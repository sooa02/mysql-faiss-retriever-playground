import os
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_mysql_db_config
from src.retrieval.hybrid_retriever import HybridRetriever

if __name__ == "__main__":
    # 캐시 경로 설정 (드라이브 내 원하는 경로)
    cache_dir = "./hf_cache"

    # 환경 변수 설정 (Hugging Face 라이브러리가 이 경로를 참조하게 함)
    os.environ['HF_HOME'] = cache_dir
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(cache_dir, "sentence_transformers")

    print("✅ 모든 AI 모델 및 데이터셋 캐시 경로가 고정되었습니다.")
    
    print("⏳ 임베딩 모델 로딩 중...")
    
    DB_CONFIG = get_mysql_db_config()
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cpu'}, # GPU 없으면 'cpu'
        encode_kwargs={'normalize_embeddings': True},
        cache_folder = os.environ['SENTENCE_TRANSFORMERS_HOME']
    )
    
    retriever = HybridRetriever(
        db_config=DB_CONFIG,
        embeddings=hf_embeddings,
        top_n=5,       
        initial_k=10,
        index_folder="data/faiss_index" #faiss 인덱스 저장 경로로 지정
    )    
    test_queries = [
        "Langchain 프로젝트를 해본 경험이 있으며, Pytorch도 사용해봤습니다."
    ]

    print("\n" + "="*50)
    print("🚀 Hybrid Retriever 테스트 시작")
    print("="*50)

    searched_selfintro = []
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
                selfintro = doc.page_content
                print(f'{idx} 번째 이력서 유사도: {doc.metadata.get("relevance_score")}')
                print(f'{idx} 번째 자소서\n{selfintro}')
                print(f'{idx} 번째 자소서 등급\n{doc.metadata.get("grade")}\n')
        except Exception as e:
            print(f"❌ 에러 발생: {e}")

    print("\n" + "="*50)
    print("✨ 모든 테스트가 종료되었습니다.")    