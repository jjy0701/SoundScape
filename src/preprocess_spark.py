import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

os.environ["HADOOP_HOME"] = "C:\hadoop"

def run_spark_pipeline():
    print("🚀 [PySpark 엔진 가동] 13900K 멀티코어 분산 처리 시작...")
    
    # 1. Spark 세션 초기화 (로컬의 모든 CPU 코어 사용: local[*])
    spark = SparkSession.builder \
        .appName("SoundScape_BigData_Preprocessing") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
    INPUT_CSV = os.path.join(DATA_DIR, 'soundscape_infinite_1m.csv')
    OUTPUT_DIR = os.path.join(DATA_DIR, 'processed_soundscape')

    print("📊 1. CSV 데이터 로드 중...")
    df = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)
    
    # 2. Feature Engineering (상대 좌표 및 거리 계산)
    print("🧬 2. 물리적 특징(Feature) 파생 변수 생성 중...")
    df = df.withColumn("dist_x", F.col("mic_x") - F.col("spk_x")) \
           .withColumn("dist_y", F.col("mic_y") - F.col("spk_y")) \
           .withColumn("distance", F.sqrt(F.col("dist_x")**2 + F.col("dist_y")**2))

    # 3. Data Leakage 방지 (방 번호 기반 Hash 분할: 8대2)
    # 💡 빅데이터 팁: room_id를 해싱(Hash)해서 나누면, 같은 방은 무조건 같은 세트로 들어갑니다.
    print("✂️  3. 데이터 누수 방지용 Train/Test 분할 중...")
    df = df.withColumn("hash_val", F.abs(F.hash(F.col("room_id"))) % 10)
    train_df = df.filter(F.col("hash_val") < 8).drop("hash_val")
    test_df = df.filter(F.col("hash_val") >= 8).drop("hash_val")

    # 4. 데이터 정규화 (StandardScaler)
    print("⚖️  4. AI 학습을 위한 데이터 정규화(Scaling) 파이프라인 구축...")
    feature_cols = ["n_vertices", "height", "rt60", "dist_x", "dist_y", "distance", "dB"]
    
    # Spark MLlib은 여러 컬럼을 하나의 '벡터'로 묶어서 처리합니다.
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)

    # 💡 핵심: 정규화 기준(평균, 표준편차)은 오직 'Train 데이터'로만 잡아야 합니다! (Test 컨닝 방지)
    pipeline = Pipeline(stages=[assembler, scaler])
    scaler_model = pipeline.fit(train_df)

    train_scaled = scaler_model.transform(train_df)
    test_scaled = scaler_model.transform(test_df)

    # 5. Parquet 포맷으로 분산 저장
    print("💾 5. 빅데이터 표준 포맷(Parquet)으로 저장 중...")
    train_scaled.write.mode("overwrite").parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
    test_scaled.write.mode("overwrite").parquet(os.path.join(OUTPUT_DIR, "test.parquet"))

    print("✅ PySpark 전처리 대장정 완료! (저장 폴더: data/processed_soundscape)")
    
    # 결과 요약 출력
    print(f"   - Train 데이터 수: {train_scaled.count():,} 건")
    print(f"   - Test 데이터 수: {test_scaled.count():,} 건")
    
    spark.stop()

if __name__ == "__main__":
    run_spark_pipeline()