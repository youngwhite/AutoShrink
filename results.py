import pandas as pd, ast, matplotlib.pyplot as plt

df = pd.read_csv('/home/usrs/wang.changlong.s8/AutoShrink/as=cifar10=vit=prune=single=0.9136(0.9007-7|12).csv')
# df = pd.read_csv('/home/usrs/wang.changlong.s8/AutoShrink/as=cifar10=vit=distill=single=0.9154(0.8978-6|12).csv')
vaccs_df = pd.DataFrame(df['vaccs'].apply(lambda x: ast.literal_eval(x)).tolist())
vaccs_df.shape
for i in range(12):
    row = vaccs_df.iloc[:, i]
    plt.plot(row, marker='.', label=f'block-{i+1}th')

plt.xlabel('Epoch (No.)')
plt.ylabel('Validation Accuracy (%)')
plt.title('Block-wise Accuracy of AutoShrink to ViT on CIFAR-10')
plt.grid()
plt.legend(loc='center right')


brow = vaccs_df.max(axis=1).idxmax()
row1 = vaccs_df.iloc[brow, :]

df = pd.read_csv('/home/usrs/wang.changlong.s8/OPMC/results/cifar10=mobilenet_v2=distill=kl=0.9248(0.9048-13|19).csv')
vaccs_df = pd.DataFrame(df['vaccs'].apply(lambda x: ast.literal_eval(x)).tolist())
brow = vaccs_df.max(axis=1).idxmax()
row2 = vaccs_df.iloc[brow, :]

# 绘制图形
plt.plot(row1.index, row1.values, label='without Self-distillation', marker='o')
plt.plot(row2.index, row2.values, label='with Self-distillation', marker='o')

# 添加标签和标题
# plt.grid()
plt.xticks(range(len(row1.index)), row1.index)
plt.xlabel('Classifier Serial Number (No.)')
plt.ylabel('Accuracy (%)')
plt.title('The Results of OPMC with & without Self-distillation')
plt.legend()


