
from dataset_tools import GarbageDatasetPreparer, ManualLabelingTool

preparer = GarbageDatasetPreparer()
preparer.create_dataset_structure()

#Put your unlabeled images into data/raw/images/ first.
labeler = ManualLabelingTool()
labeler.run()
