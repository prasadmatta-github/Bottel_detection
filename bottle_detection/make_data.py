import pandas as pd
import cv2, os
#Index(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'], dtype='object')

data = pd.read_csv('test_labels.csv')
for cnt,i in enumerate(data['filename']):
    x,y,w,h = data.iloc[cnt,:]['xmin'],data.iloc[cnt,:]['ymin'],data.iloc[cnt,:]['xmax'],data.iloc[cnt,:]['ymax']
    image = cv2.imread(os.path.join('test',i))[y:h,x:w]
    if data.iloc[cnt,:]['class']=='Fanta':
        cv2.imwrite('numbers/Fanta/{0}.jpg'.format(i),image)
    else:
        cv2.imwrite('numbers/Drpepper/{0}.jpg'.format(i),image)
