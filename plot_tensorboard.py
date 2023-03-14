import matplotlib.pyplot as plt
import numpy as np
import json
ACCURACIES_TEST_PATH = 'accuracies_test (runs_embedder_AFGRL_dataset_photo_checkpoint_dir__task_node_layers_[512]_pred_hid_1024_topk_4_num_centroids_100_num_kmeans_5_eval_freq_5_mad_0.9_lr_0.001_es_3000).json'
ERR_TEST_PATH = 'accuracies_test_err (runs_embedder_AFGRL_dataset_photo_checkpoint_dir__task_node_layers_[512]_pred_hid_1024_topk_4_num_centroids_100_num_kmeans_5_eval_freq_5_mad_0.9_lr_0.001_es_3000).json'
ACCURACIES_VALIDATION_PATH = 'accuracies_validation (runs_embedder_AFGRL_dataset_photo_checkpoint_dir__task_node_layers_[512]_pred_hid_1024_topk_4_num_centroids_100_num_kmeans_5_eval_freq_5_mad_0.9_lr_0.001_es_3000).json'
ERR_VALIDATION_PATH = 'accuracies_validation_err (runs_embedder_AFGRL_dataset_photo_checkpoint_dir__task_node_layers_[512]_pred_hid_1024_topk_4_num_centroids_100_num_kmeans_5_eval_freq_5_mad_0.9_lr_0.001_es_3000).json'

f = open(ACCURACIES_TEST_PATH)
acc_test_data = np.array(json.load(f))[:,1:]
f.close()
err_test_data = np.array(json.load(open(ERR_TEST_PATH)))[:,1:]

f = open(ACCURACIES_VALIDATION_PATH)
acc_val_data = np.array(json.load(f))[:,1:]
f.close()
err_val_data = np.array(json.load(open(ERR_VALIDATION_PATH)))[:,1:]

# print(err_test_data + acc_test_data[:,0])
# print(err_test_data.shape, acc_test_data[:,0].shape)
# print(acc_val_data[:,0])

fig, ax = plt.subplots()
ax.plot(acc_test_data[:,0], acc_test_data[:,1], color='green', label='testing accuracy')
ax.plot(acc_val_data[:,0], acc_val_data[:,1], color='blue', label='validation accuracy')
ax.fill_between(acc_test_data[:,0], acc_test_data[:,1] - err_test_data[:,1], acc_test_data[:,1] + err_test_data[:,1], alpha=0.3, color='green')
ax.fill_between(acc_val_data[:,0], acc_val_data[:,1] - err_val_data[:,1], acc_val_data[:,1] + err_val_data[:,1], alpha=0.3, color='blue')
ax.set_ylim([80,95])
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy Percentage')
ax.set_title('Testing and Validation Accuracies')
ax.legend(loc='lower right')
plt.savefig('afgrl-photo-dataset-accuracies.png')
plt.show()