import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("translations/word_model/bleu_time.csv")

# 设置图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 左边Y轴：BLEU
color = 'tab:blue'
ax1.set_xlabel('Beam Size')
ax1.set_ylabel('BLEU Score', color=color)
ax1.plot(df["beam_size"], df["bleu"], marker='o', color=color, label='BLEU Score')
ax1.tick_params(axis='y', labelcolor=color)

# 右边Y轴：Time
ax2 = ax1.twinx()  # 共享x轴
color = 'tab:green'
ax2.set_ylabel('Translation Time (seconds)', color=color)
ax2.plot(df["beam_size"], df["time"], marker='s', linestyle='--', color=color, label='Time Taken')
ax2.tick_params(axis='y', labelcolor=color)

# 图表标题
plt.title('Impact of Beam Size on BLEU Score and Translation Time')

# 图例（合并两个 y 轴的 label）
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

# 保存 & 显示
plt.tight_layout()
plt.savefig("bleu_and_time.png")
plt.show()