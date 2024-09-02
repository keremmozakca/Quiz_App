# -*- coding: utf-8 -*-

from flask import Flask,render_template,request, redirect, url_for, session
import sqlite3
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

def init_db():
    conn = sqlite3.connect('quiz.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scores
                 (username TEXT, score INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS questions
                 (id INTEGER PRIMARY KEY, question TEXT, code_snippet TEXT, option1 TEXT, option2 TEXT, option3 TEXT, option4 TEXT, correct_option TEXT)''')
    conn.commit()
    conn.close()


@app.route('/' , methods=['GET','POST'])
def home():
    if request.method=='POST':
        username = request.form.get('username')
        #bs = get_high_score(username)
        return redirect(url_for('quiz',username=username))#,best_score=bs))
    return render_template('home.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    username = request.args.get('username')

    best_score = get_high_score(username)  # En yüksek skoru alıyorum

    #print(f"Best score for {username}: {best_score}")
    
    if 'questions' not in session: #or session.get('username') != username:
        conn = sqlite3.connect('quiz.db')
        c = conn.cursor()
        c.execute('SELECT * FROM questions ORDER BY RANDOM() LIMIT 5')
        session['questions'] = c.fetchall()
        conn.close()

    questions = session['questions']

    if request.method == 'POST':
        answers = []
        for question in questions:
            answer = request.form.get(f'question{question[0]}')
            answers.append(answer)

        my_score = calculate_score(questions, answers)

        send_score_to_db(username, my_score)

        high_score = get_high_score(username)

        session.pop('questions', None)  # Oturumdan soruları kaldırıyorum

        return render_template('result.html', score=my_score, best_score=high_score, username=username)

    return render_template('quiz.html', questions=questions, username=username, best_score=best_score)


def calculate_score(q: list, a: list) -> int:
    score = 0
    #print(q) DEBUG KODU
    for question, answer in zip(q, a):
        #print(f"Question: {question[1]}, Answer: {answer}, Correct: {question[7]}") DEBUG KODU
        if answer == question[7]:  # 7. indeks doğru cevabı temsil ediyorum.
            score += 20
    return score


def send_score_to_db(username: str,score : int): #Veri tabanına puani ekleme
    conn = sqlite3.connect('quiz.db')
    c = conn.cursor()
    c.execute('INSERT INTO scores (username, score) VALUES (?, ?)', (username, score))
    conn.commit()
    conn.close()

def get_high_score(username): #En yuksek puani veritabanindan hesaplayarak cekme
    if control_new_user(username):
        conn = sqlite3.connect('quiz.db')
        c = conn.cursor()
        c.execute('SELECT MAX(score) FROM scores WHERE username=?', (username,))
        best_score = c.fetchone()[0]
        conn.close()
        return best_score if best_score is not None else 0
    return 0


def control_new_user(user):
    conn = sqlite3.connect('quiz.db')
    c = conn.cursor()
    c.execute('SELECT username FROM scores WHERE username=?', (user,))
    record = c.fetchone()
    conn.close()
    return record is not None

    for record in records:
        if(user == record[0]):
            return True
        else:
            return False


def insert_questions():
    conn = sqlite3.connect('quiz.db')
    c = conn.cursor()
    
    questions = [
    # Python'da AI Geliştirme
    (1, 'Python\'da bir yapay zeka modelinin eğitimi için hangi kütüphane yaygın olarak kullanılır?', '', 'NumPy', 'TensorFlow', 'Matplotlib', 'Requests', 'TensorFlow'),
    (2, 'Scikit-learn kütüphanesinde modelin doğruluğunu değerlendirmek için hangi metrik kullanılabilir?', '', 'F1 Skoru', 'Confusion Matrix', 'ROC AUC', 'Tüm seçenekler', 'Tüm seçenekler'),
    (3, 'Hangi Python kütüphanesi, veri ön işleme ve analiz için sıklıkla kullanılır?', '', 'Pandas', 'Flask', 'Django', 'Pygame', 'Pandas'),
    (4, 'Bir yapay zeka modelinde overfitting’i önlemek için hangi teknikler kullanılabilir?', '', 'Dropout', 'Data Augmentation', 'Regularization', 'Hepsi', 'Hepsi'),
    (5, 'Aşağıdaki kod parçacığı hangi kütüphaneye aittir?', 'from keras.models import Sequential\nfrom keras.layers import Dense', 'Keras', 'PyTorch', 'NumPy', 'SciPy', 'Keras'),

    # Bilgisayar Görüşü
    (6, 'Bilgisayar görüşü problemlerinde hangi teknik görüntü özelliklerini çıkarmak için kullanılır?', '', 'Convolutional Neural Networks (CNNs)', 'Linear Regression', 'Decision Trees', 'K-Means', 'Convolutional Neural Networks (CNNs)'),
    (7, 'OpenCV kütüphanesi hangi dili kullanır?', '', 'JavaScript', 'Python', 'C++', 'Ruby', 'Python'),
    (8, 'Hangi algoritma yüz tanıma uygulamalarında yaygın olarak kullanılır?', '', 'K-Nearest Neighbors (KNN)', 'Support Vector Machines (SVM)', 'Haar Cascades', 'Decision Trees', 'Haar Cascades'),
    (9, 'Görüntüdeki kenarları algılamak için hangi filtre yaygın olarak kullanılır?', '', 'Gaussian Filter', 'Sobel Filter', 'Median Filter', 'Laplacian Filter', 'Sobel Filter'),
    (10, 'Bir nesnenin konumunu belirlemek için hangi tekniği kullanabiliriz?', '', 'Histogram of Oriented Gradients (HOG)', 'Naive Bayes', 'Principal Component Analysis (PCA)', 'k-Means Clustering', 'Histogram of Oriented Gradients (HOG)'),

    # NLP (Nöro-Dilbilim)
    (11, 'NLP\'de dil modelleme için hangi yaklaşım kullanılır?', '', 'LSTM', 'Random Forest', 'Support Vector Machine', 'K-Means', 'LSTM'),
    (12, 'Hangi teknik metinlerdeki kelime ilişkilerini anlamak için kullanılır?', '', 'Word Embeddings', 'Convolutional Neural Networks', 'Linear Regression', 'Principal Component Analysis', 'Word Embeddings'),
    (13, 'BERT hangi tür bir modeldir?', '', 'Transformer', 'Convolutional Neural Network', 'Recurrent Neural Network', 'Decision Tree', 'Transformer'),
    (14, 'Gözlemlenen metni anlamlandırmak için kullanılan teknik nedir?', '', 'Named Entity Recognition (NER)', 'K-Means Clustering', 'Principal Component Analysis', 'Logistic Regression', 'Named Entity Recognition (NER)'),
    (15, 'Bir metindeki tüm kelimeleri saymak için hangi yöntem kullanılır?', '', 'Bag of Words', 'Support Vector Machine', 'Neural Network', 'Naive Bayes', 'Bag of Words'),

    # Python Uygulamalarında AI Modelleri
    (16, 'Aşağıdaki kod parçası hangi kütüphaneye ait bir örnektir?', 'import torch\nmodel = torch.nn.Linear(10, 2)', 'PyTorch', 'TensorFlow', 'Scikit-learn', 'Keras', 'PyTorch'),
    (17, 'Küçük veri setlerinde model performansını değerlendirmek için hangi yöntem kullanılır?', '', 'Cross-Validation', 'Grid Search', 'Data Augmentation', 'Dropout', 'Cross-Validation'),
    (18, 'Hangi Python kütüphanesi, yapay zeka modellerinin dağıtımı için kullanılır?', '', 'Flask', 'TensorFlow Serving', 'Keras', 'Pandas', 'TensorFlow Serving'),
    (19, 'Aşağıdaki kod parçası hangi model türünü temsil eder?', 'from transformers import BertTokenizer, BertModel\nmodel = BertModel.from_pretrained("bert-base-uncased")', 'Transformer', 'Convolutional Neural Network', 'Recurrent Neural Network', 'Decision Tree', 'Transformer'),
    (20, 'AI modelleri için hiperparametre optimizasyonunda hangi yöntem kullanılır?', '', 'Grid Search', 'Stochastic Gradient Descent', 'Principal Component Analysis', 'K-Nearest Neighbors', 'Grid Search')
    ]

    c.executemany('INSERT INTO questions (id, question, code_snippet ,option1, option2, option3, option4, correct_option) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', questions)
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

