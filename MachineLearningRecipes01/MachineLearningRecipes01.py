from sklearn import tree

# features = [[140g, "smooth"], [130g, "smooth"], [150g, "bumby"], [170g, "bumby"];
# labels = ["apple", "apple", "orange", "orange"]

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# print clf.predict([[160g, "bumby"]])
print(clf.predict([[160, 0]]))
