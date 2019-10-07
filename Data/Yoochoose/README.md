**This dataset is extracted from the raw yoochoose-click dataset**

Due to the large size of yoochoose-click dataset, this dataset only keeps the click sequence of sessions, and each line is a session.

And we removed items with less than 10 occurrence. After this, we removed the session with length less than 10. And we renamed the items with it's index, so it can be directly feed into embedding-lookup layer.

The modified dense dataset contains 450k sessions and 29k items.