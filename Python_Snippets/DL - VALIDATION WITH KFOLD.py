
# X variables: X_train9
# y variables: y_train2

k = 3
num_val_samples = len(X_train9) // k
num_epochs = 50
all_scores = []

for i in range(k):
    print('processing fold #', i)
    val_data = X_train9[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train2[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [X_train9[:i * num_val_samples],
         X_train9[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train2[:i * num_val_samples],
         y_train2[(i + 1) * num_val_samples:]],
        axis=0)

    model = model
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)



## show all scores:
all_scores

# average:
np.mean(all_scores)