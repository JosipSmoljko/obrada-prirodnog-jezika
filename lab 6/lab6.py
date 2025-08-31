from sklearn.linear_model import LogisticRegression
import math

# ---------- Učitavanje podataka ----------
def load_data(rec_file, class_file):
    with open(rec_file, "r") as f:
        recenzije = [line.strip().lower() for line in f.readlines()]
    with open(class_file, "r") as f:
        klase = [1 if line.strip().lower()=="pozitivno" else 0 for line in f.readlines()]
    return recenzije, klase

rec_train, klase_train = load_data("recenzijeTrain.txt", "klaseTrain.txt")
rec_test, klase_test = load_data("recenzijeTest.txt", "klaseTest.txt")

# ---------- Kreiranje rječnika ----------
rijeci = set()
for rec in rec_train:
    rijeci.update(rec.split())
rijecnik = sorted(list(rijeci))

# ---------- One-hot encoding ----------
def one_hot(rec_list, rijecnik):
    X = []
    for rec in rec_list:
        niz = [0]*len(rijecnik)
        for rijec in rec.split():
            if rijec in rijecnik:
                niz[rijecnik.index(rijec)] = 1
        X.append(niz)
    return X

X_train = one_hot(rec_train, rijecnik)
X_test = one_hot(rec_test, rijecnik)

# ---------- Logistička regresija ----------
logistic_regressor = LogisticRegression(max_iter=200)
logistic_regressor.fit(X_train, klase_train)
pred_logistic = logistic_regressor.predict(X_test)
print("Logistička regresija predikcije:", pred_logistic)

# ---------- Naivni Bayes ručno ----------
def train_naive_bayes(X, Y):
    n_rijeci = len(X[0])
    n_positiv = sum(Y)
    n_negativ = len(Y) - n_positiv
    # Laplace smoothing
    p_w_given_pos = [(sum(X[i][j] for i in range(len(X)) if Y[i]==1)+1)/(n_positiv+2) for j in range(n_rijeci)]
    p_w_given_neg = [(sum(X[i][j] for i in range(len(X)) if Y[i]==0)+1)/(n_negativ+2) for j in range(n_rijeci)]
    p_pos = n_positiv / len(Y)
    p_neg = n_negativ / len(Y)
    return p_w_given_pos, p_w_given_neg, p_pos, p_neg

def predict_naive_bayes(x, p_w_pos, p_w_neg, p_pos, p_neg):
    log_prob_pos = math.log(p_pos)
    log_prob_neg = math.log(p_neg)
    for i in range(len(x)):
        if x[i]==1:
            log_prob_pos += math.log(p_w_pos[i])
            log_prob_neg += math.log(p_w_neg[i])
        else:
            log_prob_pos += math.log(1 - p_w_pos[i])
            log_prob_neg += math.log(1 - p_w_neg[i])
    return 1 if log_prob_pos > log_prob_neg else 0

p_w_pos, p_w_neg, p_pos, p_neg = train_naive_bayes(X_train, klase_train)
pred_nb = [predict_naive_bayes(X_test[i], p_w_pos, p_w_neg, p_pos, p_neg) for i in range(len(X_test))]

# ---------- Točnost ----------
tocnost = sum(1 for i in range(len(pred_nb)) if pred_nb[i] == klase_test[i]) / len(klase_test)
print("Naivni Bayes predikcije:", pred_nb)
print(f"Točnost Naivnog Bayesa: {tocnost*100:.2f}%")