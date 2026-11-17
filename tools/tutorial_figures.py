"""Result plots appended to the six synthetic, stepwise tutorials."""

FIGURES = {
    "01_extract": (
        """import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
frame = result.features
axes[0].bar(frame['ID'], frame['original_firstorder_Mean'])
axes[0].set(xlabel='Synthetic sample', ylabel='Mean (synthetic intensity units)')
axes[1].bar(frame['ID'], frame['original_shape_basic_Volume_mm3'])
axes[1].set(xlabel='Synthetic sample', ylabel='ROI volume (mm³)')
fig.tight_layout()
fig.savefig(output/'extraction-results.png', dpi=140)
plt.close(fig)"""
    ),
    "03_multiclass_regression": (
        """import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for task in ('multiclass', 'regression'):
    saved = json.loads((output/task/'training.json').read_text(encoding='utf-8'))
    predictions = rad.PredictionTable(saved['models'][0]['predictions']).aggregate_patients().dataframe
    truth = task_data(task)[0][['patient_id', 'target']].drop_duplicates('patient_id')
    heldout = predictions.loc[predictions['group']=='test'].merge(truth, on='patient_id', validate='one_to_one')
    assert len(heldout)>0 and heldout['target'].notna().all()
    if task == 'multiclass':
        classes = task_data(task)[2]
        predicted = np.asarray(classes)[heldout[['probability::'+str(c) for c in classes]].to_numpy().argmax(axis=1)]
        confusion = pd.crosstab(heldout['target'], predicted).reindex(index=classes, columns=classes, fill_value=0)
        axes[0].imshow(confusion.to_numpy(), cmap='Blues', vmin=0)
        for y in range(len(classes)):
            for x in range(len(classes)):
                axes[0].text(x, y, str(confusion.iloc[y,x]), ha='center', va='center')
        axes[0].set(xticks=range(len(classes)), xticklabels=classes, yticks=range(len(classes)), yticklabels=classes,
                    xlabel='Predicted class', ylabel='Actual class', title='Held-out synthetic patients')
    else:
        axes[1].scatter(heldout['target'], heldout['prediction'])
        bounds = [min(heldout['target'].min(), heldout['prediction'].min()), max(heldout['target'].max(), heldout['prediction'].max())]
        axes[1].plot(bounds, bounds, linestyle='--', color='grey')
        axes[1].set(xlabel='Actual (synthetic target units)', ylabel='Predicted (synthetic target units)', title='Held-out synthetic patients')
fig.tight_layout()
fig.savefig(output/'task-results.png', dpi=140)
plt.close(fig)"""
    ),
    "04_regions_delta_habitat": (
        """import matplotlib.pyplot as plt
labels = habitat.predict(training)['cluster'].to_numpy()
fig, ax = plt.subplots(figsize=(5, 4))
for cluster in sorted(set(labels)):
    points = training.loc[labels==cluster]
    ax.scatter(points['intensity_a'], points['intensity_b'], label='Local cluster '+str(cluster))
centers = habitat.centers_
ax.scatter(centers['intensity_a'], centers['intensity_b'], marker='x', s=100, c='black', label='Saved centers')
ax.set(xlabel='Synthetic intensity a', ylabel='Synthetic intensity b', title='One local habitat fit')
ax.legend()
fig.tight_layout()
fig.savefig(output/'local-habitat.png', dpi=140)
plt.close(fig)"""
    ),
}
