import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("arpita1607067/depression-multi-class-classification-dataset")

print("Path to dataset files:", path)

print("? Files inside dataset directory:")
for file in os.listdir(path):
    print("-", file)


csv_path = os.path.join(path, "depression_dataset.csv")  # 파일명 확인 필요
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(df.head())
else:
    print("?? CSV 파일을 찾을 수 없습니다. 실제 파일 이름을 확인해주세요.")


print("? 코드 실행 성공!")

