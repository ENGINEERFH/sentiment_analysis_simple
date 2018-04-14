import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics.association import BigramAssocMeasures
from nltk.metrics.scores import precision, recall, f_measure
from nltk.probability import FreqDist, ConditionalFreqDist


POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')


# 此功能采用特征选择机制, 并以各种度量方式返回其性能
def evaluate_features(feature_select):
    posFeatures = []
    negFeatures = []
    #将这些句子分解成单词的列表（由输入机制选择）并在每个列表后附加'pos'或'neg'
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)

	
    #选择3/4用于训练和1/4用于测试
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    
    #训练朴素贝叶斯分类器
    classifier = NaiveBayesClassifier.train(trainFeatures)	

    #启动referenceSets和testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)	

    #在referenceSets中放置正确标记的句子，在测试集中放置预测性标记的版本
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)	

    #打印指标以显示特征选择的效果
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
    print 'pos F1:', f_measure(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
    print 'neg F1:', f_measure(referenceSets['pos'], testSets['pos'])
    classifier.show_most_informative_features(10) 

#创建使用所有单词的特征选择机制
def make_full_dict(words):
    return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
#print 'using all words as features'
#evaluate_features(make_full_dict)

#基于卡方检验分数词显示信息增益（http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/）
def create_word_scores():
    # 创建所有正面和负面词汇的清单
    posWords = []
    negWords = []
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
            posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords.append(posWord)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
            negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords.append(negWord)
    
    # 将多维列表转为一维列表
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    #建立所有单词的频率分布，然后建立正负标签内的单词的频率分布
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1
    #找出正面和负面词的数量，以及词的总数
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
#
    #建立基于卡方检验的单词分数字典
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores

#发现单词分数
word_scores = create_word_scores()

#根据单词分数找到最好的“数字”单词
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

#创建仅使用最佳单词的特征选择机制
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

#要选择的功能数量
numbers_to_test = [10, 100, 1000, 10000, 15000]
#尝试best_word_features机制与每个功能的numbers_to_test
for num in numbers_to_test:
    print 'evaluating best %d word features' % (num)
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)
