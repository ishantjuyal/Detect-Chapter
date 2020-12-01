# Solve

In India, every year lacs of students sit for competitive examinations like JEE Advanced, JEE Mains, NEET, etc. These exams are said to be the gateway to get admission into India's premier Institutes such as IITs, NITs, AIIMS, etc. Keeping in mind that the competition is tough as lacs of students appear for these examinations, there has been an enormous development in Ed Tech Industry in India, fortuning the dreams of lacs of aspirants via providing online as well as offline coaching, mentoring, etc. 

Solve is a project that focuses on categorising JEE questions into chapters they belong to so that appropriate solution and counselling could be provided.
The inputs required for this are:
1. The question itself
2. The class from which that chapter is.

### Dataset

Content

This particular dataset consists of questions/doubts raised by students preparing for exams mentioned above.
The dataset contains 3 CSV files. The data is split randomly across these 3 CSV files.

Inside each CSV file, we have four columns:

q_id: Questions id, unique for every question
eng: The full question or description of the questions
class: The question belongs to which class/grade in the Indian Education system.
chapter: Target classes,

So, it's basically an NLP problem where we have the question description and we need to find out which chapter does this question belongs to.

The dataset contains 200 categories. It is a massive multiclass classification task and the data is huge.

### Approach

The approach used by me was:

1. Preprocess the dataset, remove all the symbols that are not needed.
2. Remove stopwords from the questions as in this case, sentence formation is not that important but the words that we use are. So stopwords don't help us in identifying the chapter to which the question belongs to. 
3. Add the class from which the question belongs to so that it can help in narrowing down chapters. In order to do so, I concatenated special tokens like "classsix" for 6th class, "classseven" for 7th class in the beginning of the sentences. This helped a lot.
4. I used pretrained BERT model for preprocessing and trained the model further on my dataset.

### Results

Accuracy score for test dataset: 79.25%

As per the classification report, f1-score and precision score for some of the classes came out to be > 0.90 which is great considering so many classes. Obviously the data was unbalanced due to so many classes. the overall f1_score was:

|               |   precision   |     recall    |    f1-score   |
| ------------- | ------------- | ------------- | ------------- |
| macro avg     |      0.69     |      0.66     |      0.65     |
| weighted avg  |      0.79     |      0.79     |      0.79     |


### Demo

[image here]

### Dataset Reference

[Kaggle Dataset](https://www.kaggle.com/mrutyunjaybiswal/questions-chapter-classification)