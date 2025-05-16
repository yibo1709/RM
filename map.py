import matplotlib.pyplot as plt
import numpy as np

years = list(range(2017, 2025))  # 2017-2024
literature_review = [0, 0, 0, 0, 0, 0, 2, 2]  # 2023: 2, 2024: 2
model_introduction = [1, 1, 0, 1, 0, 0, 0, 0]  # 2017: 1, 2018: 1, 2020: 1
application = [0, 0, 0, 0, 0, 0, 1, 0]  # 2023: 1
limitation = [0, 0, 0, 0, 0, 0, 1, 1]  # 2023: 1, 2024: 1
future = [0, 0, 0, 0, 0, 0, 1, 1]  # 2023: 1, 2024: 1

plt.figure(figsize=(10, 6))
plt.bar(years, literature_review, label='Literature Review', color='orange')
plt.bar(years, model_introduction, bottom=literature_review, label='Model Introduction', color='yellow')
plt.bar(years, application, bottom=np.array(literature_review) + np.array(model_introduction), label='Application', color='red')
plt.bar(years, limitation, bottom=np.array(literature_review) + np.array(model_introduction) + np.array(application), label='Limitation', color='blue')
plt.bar(years, future, bottom=np.array(literature_review) + np.array(model_introduction) + np.array(application) + np.array(limitation), label='Future', color='green')

plt.title('Distribution of LLM Research Topics (2017-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Studies')
plt.xticks(years)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('llm_research_distribution_new.png')