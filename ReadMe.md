# Word to Markdown Converter

## Results of converting "Assignment 1"

### Markdown

```
Naive Bayesian and Public Opinion Analysis

1. Knowledge of the mathematical principles of Naive Bayes

Naive Bayesian is a classification method based on Bayes&#39; theorem and independent assumption of characteristic conditions. It calculates the probability of classification by features and selects the situation with high probability. It is a machine learning classification method (supervised learning) based on probability theory, which is widely used in the field of sentiment classification classifier.

1.1 What is a probability-based approach?

A measure of the likelihood of an event occurring by probability. Probability theory and statistics are two opposite concepts, statistics is to take part of the sample statistics to estimate the overall situation, while probability theory is to estimate the occurrence of a single event or part of the event through the overall situation. Probability theory requires known data to predict unknown events.

For example, we see cloudy weather, lightning, thunder, and gusts of wind. Under such weather characteristics (F), we infer that the probability of rain is greater than the probability of no rain, i.e. P (rain)\&gt; P (no rain), so we assume that it will rain later. This is an empirically judged probability. Through years of long-term accumulation of data, the weather bureau calculate the probability of rain today P (rain)=85%, P (no rain)=15%, the same P (rain)\&gt; P (no rain), so today&#39;s weather forecast definitely forecast rain. This is a method to calculate the probability of a rain event.

1.2conditional probability

If Ω is complete, A and B is one of the events (subset), conditional probability represents the probability that one event occurs when another event occurs. Suppose that the probability of event A occurring after event B is:

![](RackMultipart20210320-4-1xclwo1_html_3a56998fb5d49cd2.png)

Suppose P(A)\&gt;0，then P(AB) = P(B|A)P(A) = P(A|B)P(B).

For the event, A, B, C, if P (AB) \&gt; 0, then P (ABC) = P (A) P (B | A) P (C | AB)

If A and B are two independent events, the intersecting probability is P(A∩B) = P(A)P(B)

1.3 total probability rule

Set Ω for the sample space for trial E. A is an event of E, and Ω is dividend into B1, B2,... , Bn, and P (Bi) \&gt; 0, i = 1, 2,... , n. Then we have:

P(A) = P(AB1)+P(AB2)+...+P(ABn) = P(A|B1)P(B1)+P(A|B2)P(B2)+...+P(A|Bn)P(Bn)

The main purpose of the total probability formula is that it can decompose a complex probability calculation problem into a number of simple probability calculation problems. Finally, the final result can be obtained by applying the additivity of probability.

1.4 bayes formula

Let Ω be the sample space of experiment E, A be the event of E, if there are k mutually exclusive and finite events, that is, B1, B2, ..., Bk is a division of Ω, and P(B1)+ P(B2)+...+P(Bk)=1, P(Bi)\&gt;0 (i=1,2,...,k), then we have：

![](RackMultipart20210320-4-1xclwo1_html_decdd9e5d918416f.png)

P(A)：Probability of event A occurring

P(A∩B)：The probability that event A and event B happen together

P(A|B)：The probability of event A occurring when B occurring

Now it is known that time A has indeed occurred. To estimate the probability that it is caused by cause Bi, it can be obtained by Bayes formula.

1.5 Prior probabilities and posterior probabilities

Priori probability is the probability obtained from previous data analysis. It generally refers to the probability of occurrence of a class of things, the probability determined by historical data or subjective judgment that has not been confirmed. The posterior probability is the probability of re-correcting after the information is obtained. It is the probability of a specific thing happening under a certain condition.

1.6 Naive Bayesian classification

The Bayesian classifier predicts the probability that an object belongs to a certain category, and then predicts its category, which is constructed based on Bayes&#39; theorem. When processing large-scale data sets, Bayesian classifiers show high classification accuracy.

Suppose there are two categories：

If p1(x,y)\&gt;p2(x,y), then put into category 1

If p1(x,y)\&lt;p2(x,y), then put into category 2

Bayes&#39; theorem:

![](RackMultipart20210320-4-1xclwo1_html_3a3c40e798d56925.png)

x and y represent feature variables, ci represents classification, and p(ci|x,y) represents the probability of being classified into category ci when the features are x, y. Therefore, combining conditional probability and Bayes&#39; theorem is:

1. If p(c1|x,y)\&gt;p(c2,|x,y), then the classification should belong to category c1
2. If p(c1|x,y)\&lt;p(c2,|x,y), then the classification should belong to category c2

The biggest advantage of Bayes&#39; theorem is that the known probability can be used to calculate the unknown probability, and if it is only to compare the size of p(ci|x,y) and p(cj|x,y), you only need to know two probabilities, with the same denominator, just compare p(x,y|ci)p(ci) and p(x,y|cj)p(cj)

1.7 Advantages and disadvantages

1) Supervised learning needs to determine the objectives of classification

2) The method is not sensitive to missing data and can still be used in the case of less data

3) Can handle the classification of multiple categories

4) Applicable to nominal data

5) Sensitive to the form of input data

6) Due to the use of prior data to predict classification, errors exist

1. Simple case of naive bayes

Scikit-Learn package provides three Naive Bayesian classification algorithms:

1. Gaussian Naive Bayesian classification
2. Multinomial Naive Bayesian classification
3. Bernoulli Naive Bayesian classification

2.1 GaussianNB

Coding: sklearn.naive\_bayes.GaussianNB(priors=None)

The following six coordinate points are randomly generated. When the x and y are both positive, the corresponding class is marked as 2; when the x and y are both negative, the corresponding class is marked as 1. The code analyzed by Gauss Naive Bayesian classification is as follows:

![](RackMultipart20210320-4-1xclwo1_html_c5ebc67a9adcf0d1.png)

The prediction result of [-0.8, -1] is class 1, that is, the x and y are both negative numbers:

![](RackMultipart20210320-4-1xclwo1_html_78384ba8156fb3d2.png)

2.2 MultinomialNB

Multinomial Naive Bayesian classification mainly used for discrete feature classification, such as text classification word statistics, with the number of occurrences as the feature value.The Multinomial model will do some smoothing when calculating the prior probability P(yk) and the conditional probability P(xi|yk). The specific formula is:

![](RackMultipart20210320-4-1xclwo1_html_9825c84c4c926ee3.png)

N is the total number of samples,

k is the total number of categories,

Nyk is the number of samples with category yk,

α is the smoothing value

![](RackMultipart20210320-4-1xclwo1_html_b207486b4be07474.png)

Nyk is the number of samples in the category yk,

n is the dimension of the feature,

Nyk, xi are samples in the category yk, the value of the i-th dimension is the number of samples in xi, α is the smooth value

Coding: sklearn.naive\_bayes.MultinomialNB(alpha=1.0, fit\_prior=True, class\_prior=None)

alpha is optional, default 1.0, add Laplace/Lidstone smoothing parameter;

fit\_prior default True, indicating whether to learn prior probabilities, parameter False indicates that all class markers have the same prior probability;

class\_prior is similar to an array, The size of the array is (n\_classes,), the default is None.

2.3 BernoulliNB

Similar to Multinomiall Naive Bayes, it is also mainly used for discrete feature classification. The difference from MultinomialNB is: MultinomialNB takes the number of occurrences as the feature value, and BernoulliNB is a binary or Boolean feature.

Coding:sklearn.naive\_bayes.BernoulliNB(alpha=1.0,binarize=0.0,fit\_prior=True,class\_prior=None)

1. Naive Bayes on Chinese text public opinion analysis

Suppose you now want to determine if a message is spam. Here are the steps:

1. Split the data set into words, Chinese word segmentation technology
2. Calculate the total number of words in the sentence and determine the size of the word vector
3. Transform the words in the sentence into vectors
4. Calculate P(Ci), P(Ci|w)=P(w|Ci)P(Ci)/P(w), which represents the conditional probability that the sample is classified as Ci when the w feature appears
5. Determine the probabilities of P(w[i]C[0]) and P(w[i]C[1]). The higher probability of the two sets is the classification label

An example of Book evaluation information

3.1 Read the Data set

Suppose there are 10 book order evaluation information as shown below, and each evaluation information corresponds to a result (good and bad reviews), as shown in the figure below:

![](RackMultipart20210320-4-1xclwo1_html_2432bc159f53df97.png)

Use pandas package to read the data set:

![](RackMultipart20210320-4-1xclwo1_html_20230f6e20727836.png) ![](RackMultipart20210320-4-1xclwo1_html_8d8e1b89541340d9.png)

3.2 Chinese word segmentation and filtering stop words

Then use jieba to segment Chinese word, and the stop words:

stopwords = {}.fromkeys([&#39;，&#39;, &#39;。&#39;, &#39;！&#39;, &#39;这&#39;, &#39;我&#39;, &#39;非常&#39;])

![](RackMultipart20210320-4-1xclwo1_html_431ce49ec8d1342d.png) ![](RackMultipart20210320-4-1xclwo1_html_72a5847fca991e95.png)

3.3 Word frequency statistics

![](RackMultipart20210320-4-1xclwo1_html_92a1a97480f9893b.png)

{python 一本 书籍 优化 优秀 但是 作者 值得 写得 准备 多年 大家 好书 好评 学习 差评 建议 强烈推荐 很差 思路 技术 拥有 数据 数据分析 最差 有点 比较 混乱 火热 用心 简直 认真 误人子弟 读者 购买 还是 这么 这是 这本 退货 逻辑 难得一见}

![](RackMultipart20210320-4-1xclwo1_html_3f5daa7218e4a858.png)

3.4 Data Analysis

![](RackMultipart20210320-4-1xclwo1_html_994d476dc64771a9.png)

The output result is shown below, and you can see the two predicted values are correct. That is, &quot;一本优秀的书籍，值得读者拥有。&quot; The predicted result is favorable (category mark 1), &quot;很差，不建议买，准备退货。&quot; The result is bad review (category mark 0)

![](RackMultipart20210320-4-1xclwo1_html_2efc6ab3379610b1.png)

But there is a problem. Because the amount of data is small and not representative, and the real analysis will use massive data for public opinion analysis, the prediction results are definitely not 100% correct, but the experimental results need to be as good as possible. Finally, add a piece of code for dimensionality reduction to draw graphics, as follows:

![](RackMultipart20210320-4-1xclwo1_html_5d92ef6190f5078a.png) ![](RackMultipart20210320-4-1xclwo1_html_9ee4d002af5d65e0.png)

The output result is shown in the figure, the predicted result and the real result are the same, namely [1,1,0,0,1,0,0,1,1,0]

![](RackMultipart20210320-4-1xclwo1_html_664b083f4febac68.png)

1. The large scale Public Opinion Analysis

The code given in this project showed how to deal with a large amount of public opinions and give the scores of positive/negative information. We can take these scores into our stock price prediction model in order to judge if the model is impact by public opinions. The code is shown in this package.
```

### Rendered

Naive Bayesian and Public Opinion Analysis

1. Knowledge of the mathematical principles of Naive Bayes

Naive Bayesian is a classification method based on Bayes' theorem and independent assumption of characteristic conditions. It calculates the probability of classification by features and selects the situation with high probability. It is a machine learning classification method (supervised learning) based on probability theory, which is widely used in the field of sentiment classification classifier.

1.1 What is a probability-based approach?

A measure of the likelihood of an event occurring by probability. Probability theory and statistics are two opposite concepts, statistics is to take part of the sample statistics to estimate the overall situation, while probability theory is to estimate the occurrence of a single event or part of the event through the overall situation. Probability theory requires known data to predict unknown events.

For example, we see cloudy weather, lightning, thunder, and gusts of wind. Under such weather characteristics (F), we infer that the probability of rain is greater than the probability of no rain, i.e. P (rain)&gt; P (no rain), so we assume that it will rain later. This is an empirically judged probability. Through years of long-term accumulation of data, the weather bureau calculate the probability of rain today P (rain)=85%, P (no rain)=15%, the same P (rain)&gt; P (no rain), so today's weather forecast definitely forecast rain. This is a method to calculate the probability of a rain event.

1.2conditional probability

If Ω is complete, A and B is one of the events (subset), conditional probability represents the probability that one event occurs when another event occurs. Suppose that the probability of event A occurring after event B is:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_3a56998fb5d49cd2.png)

Suppose P(A)&gt;0，then P(AB) = P(B|A)P(A) = P(A|B)P(B).

For the event, A, B, C, if P (AB) &gt; 0, then P (ABC) = P (A) P (B | A) P (C | AB)

If A and B are two independent events, the intersecting probability is P(A∩B) = P(A)P(B)

1.3 total probability rule

Set Ω for the sample space for trial E. A is an event of E, and Ω is dividend into B1, B2,... , Bn, and P (Bi) &gt; 0, i = 1, 2,... , n. Then we have:

P(A) = P(AB1)+P(AB2)+...+P(ABn) = P(A|B1)P(B1)+P(A|B2)P(B2)+...+P(A|Bn)P(Bn)

The main purpose of the total probability formula is that it can decompose a complex probability calculation problem into a number of simple probability calculation problems. Finally, the final result can be obtained by applying the additivity of probability.

1.4 bayes formula

Let Ω be the sample space of experiment E, A be the event of E, if there are k mutually exclusive and finite events, that is, B1, B2, ..., Bk is a division of Ω, and P(B1)+ P(B2)+...+P(Bk)=1, P(Bi)&gt;0 (i=1,2,...,k), then we have：

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_decdd9e5d918416f.png)

P(A)：Probability of event A occurring

P(A∩B)：The probability that event A and event B happen together

P(A|B)：The probability of event A occurring when B occurring

Now it is known that time A has indeed occurred. To estimate the probability that it is caused by cause Bi, it can be obtained by Bayes formula.

1.5 Prior probabilities and posterior probabilities

Priori probability is the probability obtained from previous data analysis. It generally refers to the probability of occurrence of a class of things, the probability determined by historical data or subjective judgment that has not been confirmed. The posterior probability is the probability of re-correcting after the information is obtained. It is the probability of a specific thing happening under a certain condition.

1.6 Naive Bayesian classification

The Bayesian classifier predicts the probability that an object belongs to a certain category, and then predicts its category, which is constructed based on Bayes' theorem. When processing large-scale data sets, Bayesian classifiers show high classification accuracy.

Suppose there are two categories：

If p1(x,y)&gt;p2(x,y), then put into category 1

If p1(x,y)&lt;p2(x,y), then put into category 2

Bayes' theorem:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_3a3c40e798d56925.png)

x and y represent feature variables, ci represents classification, and p(ci|x,y) represents the probability of being classified into category ci when the features are x, y. Therefore, combining conditional probability and Bayes' theorem is:

1. If p(c1|x,y)&gt;p(c2,|x,y), then the classification should belong to category c1
2. If p(c1|x,y)&lt;p(c2,|x,y), then the classification should belong to category c2

The biggest advantage of Bayes' theorem is that the known probability can be used to calculate the unknown probability, and if it is only to compare the size of p(ci|x,y) and p(cj|x,y), you only need to know two probabilities, with the same denominator, just compare p(x,y|ci)p(ci) and p(x,y|cj)p(cj)

1.7 Advantages and disadvantages

1. Supervised learning needs to determine the objectives of classification
2. The method is not sensitive to missing data and can still be used in the case of less data
3. Can handle the classification of multiple categories
4. Applicable to nominal data
5. Sensitive to the form of input data
6. Due to the use of prior data to predict classification, errors exist

1. Simple case of naive bayes

Scikit-Learn package provides three Naive Bayesian classification algorithms:

1. Gaussian Naive Bayesian classification
2. Multinomial Naive Bayesian classification
3. Bernoulli Naive Bayesian classification

2.1 GaussianNB

Coding: sklearn.naive_bayes.GaussianNB(priors=None)

The following six coordinate points are randomly generated. When the x and y are both positive, the corresponding class is marked as 2; when the x and y are both negative, the corresponding class is marked as 1. The code analyzed by Gauss Naive Bayesian classification is as follows:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_c5ebc67a9adcf0d1.png)

The prediction result of [-0.8, -1] is class 1, that is, the x and y are both negative numbers:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_78384ba8156fb3d2.png)

2.2 MultinomialNB

Multinomial Naive Bayesian classification mainly used for discrete feature classification, such as text classification word statistics, with the number of occurrences as the feature value.The Multinomial model will do some smoothing when calculating the prior probability P(yk) and the conditional probability P(xi|yk). The specific formula is:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_9825c84c4c926ee3.png)

N is the total number of samples,

k is the total number of categories,

Nyk is the number of samples with category yk,

α is the smoothing value

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_b207486b4be07474.png)

Nyk is the number of samples in the category yk,

n is the dimension of the feature,

Nyk, xi are samples in the category yk, the value of the i-th dimension is the number of samples in xi, α is the smooth value

Coding: sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

alpha is optional, default 1.0, add Laplace/Lidstone smoothing parameter;

fit_prior default True, indicating whether to learn prior probabilities, parameter False indicates that all class markers have the same prior probability;

class_prior is similar to an array, The size of the array is (n_classes,), the default is None.

2.3 BernoulliNB

Similar to Multinomiall Naive Bayes, it is also mainly used for discrete feature classification. The difference from MultinomialNB is: MultinomialNB takes the number of occurrences as the feature value, and BernoulliNB is a binary or Boolean feature.

Coding:sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=0.0,fit_prior=True,class_prior=None)

1. Naive Bayes on Chinese text public opinion analysis

Suppose you now want to determine if a message is spam. Here are the steps:

1. Split the data set into words, Chinese word segmentation technology
2. Calculate the total number of words in the sentence and determine the size of the word vector
3. Transform the words in the sentence into vectors
4. Calculate P(Ci), P(Ci|w)=P(w|Ci)P(Ci)/P(w), which represents the conditional probability that the sample is classified as Ci when the w feature appears
5. Determine the probabilities of P(w[i]C[0]) and P(w[i]C[1]). The higher probability of the two sets is the classification label

An example of Book evaluation information

3.1 Read the Data set

Suppose there are 10 book order evaluation information as shown below, and each evaluation information corresponds to a result (good and bad reviews), as shown in the figure below:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_2432bc159f53df97.png)

Use pandas package to read the data set:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_20230f6e20727836.png) ![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_8d8e1b89541340d9.png)

3.2 Chinese word segmentation and filtering stop words

Then use jieba to segment Chinese word, and the stop words:

stopwords = {}.fromkeys(['，', '。', '！', '这', '我', '非常'])

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_431ce49ec8d1342d.png) ![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_72a5847fca991e95.png)

3.3 Word frequency statistics

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_92a1a97480f9893b.png)

{python 一本 书籍 优化 优秀 但是 作者 值得 写得 准备 多年 大家 好书 好评 学习 差评 建议 强烈推荐 很差 思路 技术 拥有 数据 数据分析 最差 有点 比较 混乱 火热 用心 简直 认真 误人子弟 读者 购买 还是 这么 这是 这本 退货 逻辑 难得一见}

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_3f5daa7218e4a858.png)

3.4 Data Analysis

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_994d476dc64771a9.png)

The output result is shown below, and you can see the two predicted values are correct. That is, "一本优秀的书籍，值得读者拥有。" The predicted result is favorable (category mark 1), "很差，不建议买，准备退货。" The result is bad review (category mark 0)

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_2efc6ab3379610b1.png)

But there is a problem. Because the amount of data is small and not representative, and the real analysis will use massive data for public opinion analysis, the prediction results are definitely not 100% correct, but the experimental results need to be as good as possible. Finally, add a piece of code for dimensionality reduction to draw graphics, as follows:

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_5d92ef6190f5078a.png) ![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_9ee4d002af5d65e0.png)

The output result is shown in the figure, the predicted result and the real result are the same, namely [1,1,0,0,1,0,0,1,1,0]

![img](https://word2md.com/RackMultipart20210320-4-1xclwo1_html_664b083f4febac68.png)

1. The large scale Public Opinion Analysis

The code given in this project showed how to deal with a large amount of public opinions and give the scores of positive/negative information. We can take these scores into our stock price prediction model in order to judge if the model is impact by public opinions. The code is shown in this package.

Copy markdown to clipboard

Want to [convert another document](https://word2md.com/)?

[Feedback](https://github.com/benbalter/word-to-markdown/blob/master/docs/CONTRIBUTING.md)[Source](https://github.com/benbalter/word-to-markdown)[Donate](https://www.patreon.com/benbalter)[Terms](https://word2md.com/terms/)[Privacy](https://word2md.com/privacy/)[@benbalter](https://ben.balter.com/)

<iframe id="redeviation-bs-sidebar" class="notranslate" aria-hidden="true" data-theme="default" data-pos="left" style="box-sizing: border-box; opacity: 0; pointer-events: none; position: fixed; top: 0px; left: 0px; width: 310px; max-width: none; height: 0px; z-index: 2147483646; speak: none; border: none; transform: translate3d(-310px, 0px, 0px); transition: width 0s ease 0.3s, height 0s ease 0.3s, opacity 0.3s ease 0s, transform 0.3s ease 0s; background-color: rgba(255, 255, 255, 0.8) !important; display: block !important; color: rgb(33, 37, 41); font-family: -apple-system, system-ui, &quot;segoe ui&quot;, Roboto, &quot;helvetica neue&quot;, Arial, &quot;noto sans&quot;, sans-serif, &quot;apple color emoji&quot;, &quot;segoe ui emoji&quot;, &quot;segoe ui symbol&quot;, &quot;noto color emoji&quot;; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: left; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"></iframe>