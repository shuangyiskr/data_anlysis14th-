import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import random
from collections import Counter
# Define a new color function for more colorful word clouds
def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    colors = [[4, 77, 82], [242, 85, 96], [242, 96, 85], [77, 196, 255], [255, 77, 196], [77, 255, 162], [255, 242, 0]]
    rand = random.randint(0, len(colors) - 1)
    return f"rgb({colors[rand][0]}, {colors[rand][1]}, {colors[rand][2]})"
plt.rcParams["axes.unicode_minus"]=False #解决图像中的"-"负号的乱码问题
# plt.rcParams['font.sans-serif']=['kaiti']
plt.rcParams['font.sans-serif']=['SimHei']
# Load stop words from the file

stopwords_path = 'cn_stopword.txt'
with open(stopwords_path, 'r', encoding='utf-8') as file:
    stopwords = set([line.strip() for line in file.readlines()])
#print(stopwords)
# Load the cleaned comments
cleaned_comments_path = 'cleaned_小红书评论.txt'
with open(cleaned_comments_path, 'r', encoding='utf-8') as file:
    cleaned_comments = file.read()

#stopwords.add('姐妹')
stopwords.add('吊')
# Load the contour image
image_path = '1.2.png'

mask_image = np.array(Image.open(image_path))

import jieba

# First we need to split the words using jieba
jieba_result = jieba.cut(cleaned_comments, cut_all=False)
print(jieba_result)
# Join the splitted words back into a string with space-separated words
split_text = " ".join(jieba_result)
print(split_text)
# 使用jieba进行分词
words = jieba.lcut(cleaned_comments)

# 进行词频统计
word_counts = Counter(words)
# 对词频进行排序，取出前20个最高频的词
top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:60]


top_words.pop(0)
top_words.pop(0)
print(top_words)

# Initialize word cloud object
wc = WordCloud(
    font_path='simkai.ttf',
    background_color='white',
    max_words=100,
    mask=mask_image,
    stopwords=stopwords,
    #contour_width=1,
    #contour_color='firebrick',
    random_state=42  # Ensure reproducibility
)

# Generate word cloud
wc.generate(split_text)

# Extract color from the image
#image_colors = ImageColorGenerator()

# Recolor the word cloud with image color
wc.recolor(color_func=random_color_func)

# Plot the word cloud
plt.figure(figsize=(8, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()