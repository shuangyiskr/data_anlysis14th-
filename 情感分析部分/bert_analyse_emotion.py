from cemotion import Cemotion
import pandas as pd
file_path = 'cleaned_小红书评论.txt'

# Open and read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    comments_text = file.read()

# For demonstration, let's just use the first few comments for sentiment analysis to avoid long processing time.
# Split the text into individual comments based on a known pattern (e.g., new lines)
comments = comments_text.split('\n')
# print(comments)
c = Cemotion()
data = c.predict(comments)
print(data)

df = pd.DataFrame(data, columns=['评论', '得分'])
df.to_excel('output.xlsx', index=False)