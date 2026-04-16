import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool
import json
import numpy as np
#import faiss
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langsmith import traceable # LangSmith 추적용
from pydantic import ConfigDict

class HybridRetriever(BaseRetriever):
    db_config: dict
    embeddings: any
    top_n: int = 3
    initial_k: int = 50
    vectorstore: FAISS = None
    db_pool: MySQLConnectionPool = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _build_index(self):
        """서버 기동 시 비정규화된 벡터를 가져와 메모리 정규화 후 FAISS 빌드"""
        conn = db_pool.get_connection() if self.db_pool is not None else mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True) # DictCursor 역할

        try:
            # 1. 분리된 두 테이블을 Join하여 ID, 등급, 벡터 로드
            # MYSQL은 VECTOR_TO_STRING을 사용하여 바이너리 데이터를 문자열로 변환
            #TiDB는 VEC_AS_TEXT 사용
            sql = """
                SELECT r.id, r.grade, VEC_AS_TEXT(v.embedding) as embedding_str
                FROM application_records r
                JOIN application_vectors v ON r.id = v.record_id
            """
            cursor.execute(sql)
            rows = cursor.fetchall()

            if not rows:
                print("⚠️ DB에 적재된 데이터가 없습니다.")
                return

            metadatas = []
            vectors = []

            for r in rows:
                metadatas.append({
                    "db_id": r['id'],
                    "grade": r['grade']
                })
                # JSON 문자열인 벡터를 파싱
                vec = json.loads(r['embedding_str']) if r['embedding_str'] else [0.0]*1024
                vectors.append(vec)

            vectors_np = np.array(vectors).astype('float32')
            #faiss.normalize_L2(vectors_np) # 이미 정규화되어 있음

            # 3. FAISS 인덱스 생성 (LangChain 인터페이스 활용)
            # text_embeddings는 (ID문자열, 벡터배열) 튜플 리스트여야 함
            text_embeddings = list(zip([str(m['db_id']) for m in metadatas], vectors_np))

            # 내적(MAX_INNER_PRODUCT)을 거리 전략으로 선택 (정규화된 벡터에서는 코사인 유사도와 동일)
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embeddings,
                metadatas=metadatas,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT #이미 정규화되어 있음 아닌경우 CONSINE
            )
            print(f"✅ FAISS 빌드 완료: {len(rows)}건 정규화 및 인덱싱 성공")

        except Error as e:
            print(f"❌ DB 연결 오류: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

    @traceable(name="Vector Search", process_inputs=lambda x: {}, process_outputs=lambda x: {})
    def _get_relevant_documents(self, query: str) -> List[Document]:
        if self.vectorstore is None: self._build_index()

        # 1. FAISS 유사도 검색 (상위 initial_k개 후보 추출)
        # LangChain의 similarity_search_with_score는 내부적으로 쿼리 벡터를 정규화하여 검색함
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.initial_k)

        score_map = {doc.metadata['db_id']: float(score) for doc, score in docs_and_scores}
        
        # 2. Peer-First (Grade 기반) 필터링: 상 -> 중 순서로 정렬
        candidates = [doc for doc, score in docs_and_scores]
        high_grade = [d for d in candidates if d.metadata['grade'] == 'high']
        mid_grade = [d for d in candidates if d.metadata['grade'] == 'mid']

        # 최종적으로 우리가 원하는 개수(top_n)만큼 ID 선별
        final_candidate_docs = (high_grade + mid_grade)[:self.top_n]
        target_db_ids = [d.metadata['db_id'] for d in final_candidate_docs]

        # 3. MySQL에서 '진짜 자소서 본문' 페치
        return self._fetch_final_documents(target_db_ids, score_map)

    def _fetch_final_documents(self, db_ids: List[int], score_map: Dict[int, float]) -> List[Document]:
        if not db_ids: return []

        conn = db_pool.get_connection() if self.db_pool is not None else mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True)

        try:
            format_strings = ','.join(['%s'] * len(db_ids))
            # content_json(원본)과 grade를 가져옴
            sql = f"SELECT id, grade, resume_cleaned FROM application_records WHERE id IN ({format_strings})"
            cursor.execute(sql, tuple(db_ids))
            rows = cursor.fetchall()

            id_map = {r['id']: r for r in rows}

            # 원래 유사도 순서(db_ids)를 유지하며 Document 객체 생성
            return [
                Document(
                    page_content=json.dumps(id_map[db_id]['resume_cleaned'], ensure_ascii=False),
                    metadata={
                        "id": db_id,
                        "grade": id_map[db_id]['grade'],
                        "score": score_map.get(db_id)
                        }
                ) for db_id in db_ids if db_id in id_map
            ]
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    from src.config import get_db_config
    from langchain_huggingface import HuggingFaceEmbeddings
    import os
    from mysql.connector import pooling
    
    DB_CONFIG = get_db_config()
    
    db_pool = pooling.MySQLConnectionPool(
    pool_name="mysql_db_pool",
    pool_size=3, # 동시 접속자 수에 따라 조절
    **DB_CONFIG
    )
    # 임베딩 모델 로드 (Qwen3-Embedding)
    print("⏳ 임베딩 모델 로딩 중...")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cuda'}, # GPU 없으면 'cpu'
        encode_kwargs={'normalize_embeddings': True},
        cache_folder = os.environ['SENTENCE_TRANSFORMERS_HOME']
    )

    # 리트리버 객체 생성
    retriever = HybridRetriever(
        db_config=DB_CONFIG,
        embeddings=hf_embeddings,
        top_n=3,       # 최종 3개 결과 도출
        initial_k=50   # FAISS에서 먼저 뽑을 후보군 수
    )

    # 검색 테스트 시나리오
    test_queries = [
        "Langchain과 Langgraph 프로젝트를 해본 경험이 있으며, Pytorch도 많이 사용해봤습니다."
    ]

    print("\n" + "="*50)
    print("🚀 Hybrid Retriever 테스트 시작")
    print("="*50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 [테스트 {i}] 질문: {query}\n")

        try:
            # 리트리버 실행 (invoke 사용)
            results = retriever.invoke(query)

            if not results:
                print("⚠️ 검색 결과가 없습니다.")
                continue

            # 결과 출력
            for idx, doc in enumerate(results, 1):
                resume= doc.page_content
                resume = json.loads(resume)
                print(f'{idx} 번째 이력서 유사도: {doc.metadata.get('score')}')
                print(f'{idx} 번째 이력서\n{resume}')
                print(f'{idx} 번째 자소서 등급\n{doc.metadata.get('grade')}\n')



        except Exception as e:
            print(f"❌ 에러 발생: {e}")

    print("\n" + "="*50)
    print("✨ 모든 테스트가 종료되었습니다.")    
