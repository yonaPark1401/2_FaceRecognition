# task 2:
import face_recognition as fr
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# my_img = fr.load_image_file('my_pic.jpg')
# my_img_encoding = fr.face_encodings(my_img)[0]
# np.save('my_face_encodings', my_img_encoding)

my_face_encodings = np.load('my_face_encodings.npy')

def compare_faces(face_to_compare):
    encodings = fr.face_encodings(face_to_compare)[0]
    compare = fr.compare_faces([my_face_encodings], encodings, tolerance=0.5)
    return compare[0]

def vectors_angle(face_img):
    vector_1 = np.array(my_face_encodings).reshape(-1, 128)
    vector_2 = np.array(fr.face_encodings(face_img)[0]).reshape(-1, 128)
    # angle = np.dot(vector_1, vector_2)/(norm(vector_1)*norm(vector_2))
    angle = cos_sim(vector_1, vector_2)
    return angle

result = []
angles = []
distances = []

imgs_path = Path('imgs').glob('*.jpg')

for path in imgs_path:
    face_img = fr.load_image_file(path)
    result.append(compare_faces(face_img))
    angles.append(vectors_angle(face_img))
print(result)
print(angles)
















