# import pandas as pd
# import matplotlib.pyplot as plt

# # Load your logged LLM selection data
# df = pd.read_csv("llm_outputs.csv")

# # Count how many times each LLM was selected per language
# counts = df.groupby(["language", "action"]).size().reset_index(name="count")

# print("\n=== Language-wise LLM Selection Counts ===")
# print(counts)

# # Pivot for visualization
# pivot = counts.pivot(index="language", columns="action", values="count").fillna(0)

# print("\n=== Pivot Table (language × LLM) ===")
# print(pivot)

# # Plot the bar chart
# pivot.plot(kind="bar", figsize=(10, 6))
# plt.title("Language-wise LLM Selection Frequency")
# plt.xlabel("Language")
# plt.ylabel("Number of Selections")
# plt.legend(title="LLM Action ID")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()
import matplotlib
print(matplotlib.get_backend())