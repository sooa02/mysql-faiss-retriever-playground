import os

from src.loader.data_loader import get_dataset

from src.preprocessing.company_cleaner import CompanyNameCleaner
from src.preprocessing.data_processor import DataProcessor
from src.preprocessing.cleaner_config import en_to_ko_map, typo_fix_map, conflict_groups, protected_keywords

from src.config import get_mysql_db_config
from src.database.bulk_loader import JobPocketBulkLoader
from src.database.ingestion_pipeline import JobPocketPipeline

if __name__ == "__main__":
    # 캐시 경로 설정 (드라이브 내 원하는 경로)
    cache_dir = "./hf_cache"

    # cache 경로 설정
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "huggingface/datasets")

    print("✅ 모든 AI 모델 및 데이터셋 캐시 경로가 고정되었습니다.")
    
    # 1. Huggingface에서 dataset 불러오기
    train_data = get_dataset('train')
    
    # 2. 데이터 전처리 관련 객체 생성
    cleaner = CompanyNameCleaner(
        en_to_ko_map = en_to_ko_map,
        typo_fix_map = typo_fix_map,
        conflict_groups = conflict_groups,
        protected_keywords = protected_keywords
    )
    processor = DataProcessor(cleaner)
    
    # 3. 데이터 전처리 파이프라인 실행
    train_df = processor.run_preprocess_pipeline(train_data)
    
    print(f"✅ 데이터셋 전처리가 완료되었습니다. 데이터 개수: {len(train_df)}")
    
    # 4. MySQL DB에 데이터 적재
    DB_CONFIG = get_mysql_db_config()
    loader = JobPocketBulkLoader(**DB_CONFIG)
    pipeline = JobPocketPipeline(
        loader = loader,
        checkpoint_file = 'checkpoint.json'
    )
    
    try:
        print("🚀 [START] 데이터 적재 파이프라인을 가동합니다.")
        pipeline.execute(train_df, chunk_size=16) # VRAM 8GB 기준
        print("🏁 [FINISH] 모든 작업이 성공적으로 종료되었습니다.")
        
    except Exception as e:
        print(f"❗ 작업 중 오류 발생: {e}")
        print("💡 체크포인트가 저장되었습니다. 문제를 해결한 후 다시 실행하면 중단 지점부터 재개합니다.")