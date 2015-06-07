def plot_precision_recall_n(y_true, y_prob, model_name):
	# Weird because precision & recall curves have n_thresholds + 1 values. Their last values are fixed so throw away last value.
	y_score = y_prob
	from sklearn.metrics import precision_recall_curve
	precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
	precision_curve = precision_curve[:-1]
	recall_curve = recall_curve[:-1]

	pct_above_per_thresh = []
	number_scored = len(y_score)
	for value in pr_thresholds:
		num_above_thresh = len(y_score[y_score>=value])
		pct_above_thresh = num_above_thresh / float(number_scored)
		pct_above_per_thresh.append(pct_above_thresh)
	pct_above_per_thresh = np.array(pct_above_per_thresh)
	plt.clf()
	fig, ax1 = plt.subplots()
	ax1.plot(pct_above_per_thresh, precision_curve, 'b')
	ax1.set_xlabel('percent of population')
	ax1.set_ylabel('precision', color='b')
	ax2 = ax1.twinx()
	ax2.plot(pct_above_per_thresh, recall_curve, 'r')
	ax2.set_ylabel('recall', color='r')
	fig.show()
	name = model_name + " Precision Recall vs Population"+ ".png"
	plt.title(name)
	plt.savefig(name)
	return fig