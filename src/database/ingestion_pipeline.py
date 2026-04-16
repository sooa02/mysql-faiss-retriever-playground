import json
import os
import hashlib
import pandas as pd
from tqdm import tqdm
from langsmith import traceable
from langchain_huggingface import HuggingFaceEmbeddings

class JobPocketPipeline:
    def __init__(self, loader, model_name="Qwen/Qwen3-Embedding-0.6B", checkpoint_file="upload_checkpoint.json"):
        self.loader = loader
        self.checkpoint_file = checkpoint_file
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.state = self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"companies": False, "jobposts": False, "applicants_last_index": 0}

    def _save_checkpoint(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f)

    @traceable(name="JobPocket_Master_Pipeline")
    def execute(self, df: pd.DataFrame, chunk_size=50):
        print("🆔 [Step 1] 관계형 ID 및 해시 생성 중...")

        # 1. Company ID 생성
        df['company_id'], _ = pd.factorize(df['company'])
        df['company_id'] += 1

        # 2. JobPost ID 생성 (description_hash를 활용해 정밀 매핑)
        def get_hash(text): return hashlib.md5(str(text).strip().encode()).hexdigest()
        df['desc_hash'] = df['description'].apply(get_hash)

        jobpost_cols = ['company', 'position_type', 'career_type', 'desc_hash']
        df['jobpost_id'] = df.groupby(jobpost_cols, sort=False).ngroup() + 1

        # 3. Applicant ID 생성
        df['applicant_id'] = range(1, len(df) + 1)

        # 🏢 [Phase 1] 기초 정보 적재 (Loader 호출)
        if not self.state["companies"]:
            self.loader.upload_companies(df)
            self.state["companies"] = True
            self._save_checkpoint()

        if not self.state["jobposts"]:
            self.loader.upload_jobposts(df)
            self.state["jobposts"] = True
            self._save_checkpoint()

        # 🚀 [Phase 2] 실시간 임베딩 생성 및 메인 데이터 적재
        total_rows = len(df)
        start_idx = self.state["applicants_last_index"]

        if start_idx < total_rows:
            with tqdm(total=total_rows, initial=start_idx, desc="Embedding & Ingesting", unit="row") as pbar:
                for i in range(start_idx, total_rows, chunk_size):
                    end_idx = min(i + chunk_size, total_rows)
                    chunk_df = df.iloc[i:end_idx].copy()

                    # 실시간 임베딩 생성하여 컬럼 추가 (Loader가 사용할 재료)
                    chunk_df['resume_embedding'] = self.embeddings.embed_documents(chunk_df['resume_cleaned'].tolist())

                #   chunk_df['selfintro_embedding'] = self.embeddings.embed_documents(chunk_df['selfintro'].tolist())

                    # loader 호출
                    self.loader.upload_applicants_and_vectors(chunk_df)

                    # 체크포인트 갱신
                    self.state["applicants_last_index"] = end_idx
                    self._save_checkpoint()
                    pbar.update(len(chunk_df))

        print("\n🎉 모든 데이터 적재가 완료되었습니다, Gloveman님!")