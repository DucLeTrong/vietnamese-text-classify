from process_data import *
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    root_path = ''
    model = create_classifier()
    X_data, y_data, X_test, y_test = preprocess_data(root_path = '')
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.05, random_state=2019)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=512)
    
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)
    
    val_predictions = val_predictions.argmax(axis=-1)
    test_predictions = test_predictions.argmax(axis=-1)
    train_predictions = train_predictions.argmax(axis=-1)

    print("Train accuract", metrics.accuracy_score(train_predictions, y_train))
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))
    
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")
    print("Saved model to disk")
   

