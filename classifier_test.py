from app.ml.classifier import WasteClassifier

clf = WasteClassifier()
print(clf.predict("old paint can"))
print(clf.predict("banana peel"))
print(clf.predict("plastic water bottle"))
print(clf.predict("onion peel"))
print(clf.predict("plastic cover"))