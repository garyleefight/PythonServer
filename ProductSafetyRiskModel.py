import nltk
nltk.download("stopwords")

import pandas as pd
import numpy as np
import json
import re
import xlsxwriter
from langdetect import detect
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_union
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import metrics
import random
import pickle
import tornado.ioloop
import tornado.httpserver
import tornado.web

# Web application that routes requests to proper handlers
class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler)
        ]
        settings = dict(
        )
        super(Application, self).__init__(handlers, **settings)

# # Language detection, and drop rows with unsupported languages
def detectLanguage(sentence):
    try:
        return detect(str(sentence)) # change this into the right format for use later one
    except Exception:
        return 'unsupported'

def drop_column(dataframe, column_name):
    return dataframe.drop([column_name], inplace=True, axis=1)

# Drop unsupported language columns before removing special characters in name
def dropRowsWithUnsupportedLanguage(dataframe, columnName):
    rowsToBeDropped = []
    for index, row in dataframe[columnName].iteritems():
        language = detectLanguage(row)
        if language == 'unsupported':
            rowsToBeDropped.append(index)
            print(index)
            print(row)
    print(rowsToBeDropped)
    temp_data_frame = dataframe.drop(dataframe.index[rowsToBeDropped], axis=0) #dataframe.drop(rowsToBeDropped, axis=1)
    return temp_data_frame

# # Removing special characters
# Replacing the special characters with space, evaluate with and without. Sometimes having special characters makes sense
# Eg: iPhone-6S
def removeSpecialCharacters(sentence):
    return re.sub('[^a-zA-Z0-9 \n\.]', ' ', sentence)

# # Dataframe column data extractor
class DataFrameColumnExtracter(TransformerMixin, BaseEstimator):

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class MainHandler(tornado.web.RequestHandler):
    def post(self):
        # training_file = self.get_argument("training_file")
        results_file = self.get_argument("results_file")
        test_file = self.get_argument("test_file")
        # product_safety_df = pd.read_excel(training_file)
        # df_with_target_label = product_safety_df.dropna(subset=['SEVERITY'])    # Drop rows without target label SEVERITY value
        # list_of_unique_asins = pd.unique(df_with_target_label['ASIN'])  # Grouping Risk levels by ASINs and assigning highest seen RISK level
        # df_with_target_label.reset_index(drop=True, inplace=True) # TODO use iloc instead of reset_index when optimizing

        # for asin in list_of_unique_asins:
        #     temp_df = df_with_target_label[df_with_target_label['ASIN'] == asin]
        #     temp_list_of_severity = pd.unique(temp_df['SEVERITY'])
        #     indexes_to_change = []
        #     if len(temp_list_of_severity) > 1:
        #         indexes_to_change = temp_df.index.tolist()
        #         if 'HIGH' in temp_list_of_severity:
        #             df_with_target_label.set_value(indexes_to_change, 'SEVERITY', 'HIGH')
        
        #         elif 'MEDIUM' in temp_list_of_severity:
        #             df_with_target_label.set_value(indexes_to_change, 'SEVERITY', 'MEDIUM')
        

        # # Remove unwaunusable columns in the current input dataset
        # drop_column(df_with_target_label, "ASIN") # Not of any use as this is a unique variable and is a unique identifier.
        
        # drop_column(df_with_target_label, "Brand") # Brand has so many values as zero and was told might not be right once by the business team.
        
        # # Item name has 137 zeroes, These 137 will be dropped later on as this is one of the most important attribute to be used 
        # # to determine the risk level.
        
        # drop_column(df_with_target_label, "REVIEW_ID") # Review ID is a unique ID got from a database export.
        
        # drop_column(df_with_target_label, "AUTHOR_ID")
        # # Is AUTHOR_ID, related to customer ID of a review or is it something populated only for books?
        # # Or the person who reviewed? Dropping for now, as nothing can be related.
        
        # #Cant say surely as we are not performing account level but only on ASIN level
        
        # drop_column(df_with_target_label, "SCORE")# Score 12138 empty columns  and rest say 0.0 - removing as of no value.
        
        # drop_column(df_with_target_label, "TT_NUMBER")# TT number - of no use to us, just a unique identifer.
        
        # drop_column(df_with_target_label, "PSDB_ID") # Remove as not useful
        
        # #Item Type - 503 empty , might have to remove them as this would be a very important feature.
        
        # drop_column(df_with_target_label, "Sub_Category") # Sub Category is empty, removing column.
        
        # # drop_column(df_with_target_label, "Country of Origin") 
        # # Build multiple models with and without removing this. More than 90% of the column is empty, but there is variation in the rest 10%
        
        # drop_column(df_with_target_label, "CCR") # Not a ASIN level attribute but is a seller level attribute. Drop it for now. 
        # # Future models built not only on product level attributes can use it.
        
        # drop_column(df_with_target_label, "Rating")  #Might have to remove this as its a,
        # # combination of product level and review level data. Its investigation/review level attribute and not product level attribute.
        
        # drop_column(df_with_target_label, "Seller ID")   # Seller ID is not of any use to us as its a seller level attribute. 
        # # Can be used if we are building seller level models later on.
        
        # drop_column(df_with_target_label, "vulnerable Score")  #vulnerable score, It's generated based on other attributes(See original BRD draft version 1)
        # # Not product level attribute
        
        # drop_column(df_with_target_label, "Rating Score")  #Rating score , It's generated based on other attributes(See original BRD draft version 1)
        # # Not product level attribute
        
        # drop_column(df_with_target_label, "Country Score")  # Country Score, It's generated based on other attributes(See original BRD draft version 1)
        # # Not product level attribute
        
        # drop_column(df_with_target_label, "Severity Score ")  #Severity Score, It's generated based on other attributes(See original BRD draft version 1)
        # # Not product level attribute
        
        # drop_column(df_with_target_label, "CCR Score")  # CCR Score, ignore it. This columns is a seller level attribute and not 
        # # for the an ASIN as such.
        
        # drop_column(df_with_target_label, "Total Score")  # Total score , ignore it. Just a sum of all other scores. The attribute 
        # # was shared based on the original BRD draft version 1.
        
        # drop_column(df_with_target_label, "PSDB_TABLE")  # Not useful, ignore it. PSDB_TABLE, not a product level attribute and is
        # # just a unique identifier.
        
        # drop_column(df_with_target_label, "MARKETPLACE_ID")  # No variation in the input date. All of them are from NA. ie 1.0 is 12754 or 7.0 is only 485
        # # What is market place ID - 0 - actually only 2, so its ok. - remove them before building model. How do you group them? So dropping.
        # # Can be an important attribute when we get ASINs from different marketplaces across regions. For now we have only NA(US,CA).
        
        # drop_column(df_with_target_label, "DATE_OF_CONCERN") # Not enough data for time series analysis. 
        # # Cannot perform was this ASINS recalled today and when will it be recalled next as not enough data.
        
        # drop_column(df_with_target_label, "INJURY") # Dropping it as it is a review/investigation level attribute but not a product level attribute.
        # # Also What does 0, 1, N, Y mean for INJURY column - can be used later when building models which want not just ASIN level attributes.
        
        # drop_column(df_with_target_label, "ID") # ID is a unique id and is of no use to us.
        
        
        # df_with_target_label["Retail or not retail"].fillna(value="Not_Retail", inplace=True)
        
        
        # # TODO Use iloc instead of reset_index while optimizing
        # df_with_target_label.reset_index(drop=True, inplace=True)
        
        
        # df_with_target_label = dropRowsWithUnsupportedLanguage(df_with_target_label,'Item Name')
        
        # #print('\n\n\n\nDF')
        # #print(df_with_target_label)
        
        
        
        
        # # Removing special characters is not helping the model being built, optional.
        
        # for index, row in df_with_target_label.iterrows():
        #     print(row['Item Name'])
        #     row['Item Name'] = removeSpecialCharacters(str(row['Item Name']))
        #     df_with_target_label.set_value(index, 'Item Name', removeSpecialCharacters(str(row['Item Name'])))
        #     print(df_with_target_label.get_value(index, 'Item Name', takeable=False))
        
        
        # # # Perform stemming and lemmatization on the text columns to get to the root word. 
        # #Caveats, if Chinese is in text then stemming wont happen. Snowball stemmer doesnt have python libraries. 
        
        # # For this to run you need to install nltk stopwords
        # stemmer = SnowballStemmer("english", ignore_stopwords=True)
        
        
        # # Stemming
        # for index, row in df_with_target_label.iterrows():
        #     print(row['Item Name'])
        #     df_with_target_label.set_value(index, 'Item Name', ",".join([ stemmer.stem(kw) for kw in df_with_target_label.get_value(index, 'Item Name', takeable=False).split(" ")]))
        #     print(df_with_target_label.get_value(index, 'Item Name', takeable=False))
        
        # # Do not use str to convert from unicode to encoded text / bytes.
        
        # # As columns are text fill missing values and junk values with 'missing' string.
        # df_with_target_label["Merchant Brand Name"].fillna(value="missing", inplace=True)
        # df_with_target_label["Item_type"].fillna(value="missing", inplace=True)
        # df_with_target_label["Country of Origin"].fillna(value="missing", inplace=True)
        
        
        
        # for index, row in df_with_target_label.iterrows():
        #     if(type(row['Item_type']) == float or type(row['Item_type']) == int or type(row['Item_type']) == bool):
        #         print(row['Item_type'])
        #         df_with_target_label.set_value(index, 'Item_type', 'missing')
        #         print(df_with_target_label.get_value(index, 'Item_type', takeable=False))
        
        
        
        # for index, row in df_with_target_label.iterrows():
        #     if(type(row['Merchant Brand Name']) == float or type(row['Merchant Brand Name']) == int or type(row['Merchant Brand Name']) == bool):
        #         print(row['Merchant Brand Name'])
        #         df_with_target_label.set_value(index, 'Merchant Brand Name', 'missing')
        #         print(df_with_target_label.get_value(index, 'Merchant Brand Name', takeable=False))
        
        
        # for index, row in df_with_target_label.iterrows():
        #     if(type(row['Country of Origin']) == float or type(row['Country of Origin']) == int or type(row['Country of Origin']) == bool):
        #         print(row['Country of Origin'])
        #         df_with_target_label.set_value(index, 'Country of Origin', 'missing')
        #         print(df_with_target_label.get_value(index, 'Country of Origin', takeable=False))
        
        # # It has int float and bool values too. Parts of dataset is not clean
        
        
        # # # Train Test split of dataset
        
        # target = df_with_target_label.pop('SEVERITY')
        # X_train, X_test, y_train, y_test = train_test_split(df_with_target_label, target, test_size=0.20, random_state=42)
        
        
        # # # Build FeatureUnion 
        
        # item_name_pipe = Pipeline([
        #        ('extractor',DataFrameColumnExtracter('Item Name')), 
        #        ('count_vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        #     ('tf_idf', TfidfTransformer(use_idf=True))
        # ])
        
        # merchant_brand_name_pipe = Pipeline([
        #        ('extractor',DataFrameColumnExtracter('Merchant Brand Name')),
        #        ('count_vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        #     ('tf_idf', TfidfTransformer(use_idf=True))
        # ])
        
        # retail_or_not_pipe = Pipeline([
        #        ('extractor',DataFrameColumnExtracter('Retail or not retail')),
        #        ('count_vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        #     ('tf_idf', TfidfTransformer(use_idf=True))
        # ])
        
        # item_type_pipe = Pipeline([
        #        ('extractor',DataFrameColumnExtracter('Item_type')),
        #        ('count_vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        #     ('tf_idf', TfidfTransformer(use_idf=True))
        # ])
        
        # country_of_origin_pipe = Pipeline([
        #        ('extractor',DataFrameColumnExtracter('Country of Origin')),
        #        ('count_vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        #     ('tf_idf', TfidfTransformer(use_idf=True))
        # ])
        
        # # Add more features as and when more data and attributes are available. Currently all chosen columns are text features
        # # and CountVectorizer should suffice. Having length of Item Name column as a feature doesn't make sense in our case.
        
        # feature_union = make_union(item_name_pipe, merchant_brand_name_pipe, retail_or_not_pipe, item_type_pipe, country_of_origin_pipe)
        # feature_union.fit_transform(X_train)
        
        
        # # # Define pipeline for easier runs
        # pipeline = Pipeline([
        #     # Use FeatureUnion to combine the features
        #     ('union', feature_union),
        
        #     # Classifier
        #     ('clf', RandomForestClassifier(n_estimators=500, max_depth=None,min_samples_split=2, random_state=42)),
        # ])
        
        
        # # # Training
        
        # pipeline.fit(X_train, y_train)
        
        
        # # Save fitted pipeline for later use, to make predictions on new data directly without entire notebook run
        
        
        filename = '/Users/lisirui/Desktop/PytonF/builtModels/fitted_model.ser'
        
        # with open(filename, 'wb') as f:
        #     pickle.dump(pipeline, open(filename, 'wb'))
        
        
        with open(filename, 'rb') as f:
          pipeline = pickle.load(f)
        
        print("Training Data Loaded!")
        # Evalauate test set
        # predicted = pipeline.predict(X_test)
        # predicted_proba = pipeline.predict_proba(X_test)
        # score = pipeline.score(X_test,y_test)
        
        
        # # # Metrics - (run for each algorithm)
        # np.mean(predicted == y_test)
        
        # print(metrics.classification_report(y_test, predicted,
        #     target_names=['LOW','MEDIUM','HIGH']))
        
        '''
        indices_to_generate_results = X_test.index.values
        
        export_excel_file = X_test.loc[indices_to_generate_results]
        
        
        export_excel_file['PREDICTED'] = pd.Series(predicted, index=export_excel_file.index)
        export_excel_file['prediction-HIGH'] = predicted_proba[:,0]
        export_excel_file['prediction-LOW'] = predicted_proba[:,1]
        export_excel_file['prediction-MEDIUM'] = predicted_proba[:,2]
        
        
        # Save results file
        
        writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
        
        # Convert the dataframe to an XlsxWriter Excel object.
        export_excel_file.to_excel(writer, sheet_name='Sheet1')
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        
        
        
        toReturn = {}
        i = 0
        for row in X_test['Item Name'].iteritems():
            toReturn[row] = '{} [{}]'.format(predicted[i], predicted_proba[i])
            i += 1
        
        print(toReturn)
        '''
        
        # # Evaluating new test data
        
        test_product_safety_df = pd.read_excel(test_file)
        
        test_risky_product_safety_df = pd.read_excel(results_file)
        
        # Verify ASINS order are same both the DF. One shared by business team and one shared by IS_RISKY/NOT team.
        test_product_safety_df = pd.concat([test_product_safety_df, test_risky_product_safety_df['IS_RISKY']], axis=1)

        asin_column = test_product_safety_df['ASIN']

        
        test_product_safety_df = test_product_safety_df[test_product_safety_df['IS_RISKY'] == 'Y']    # Only use ASINs marked as risky 
        
        
        test_product_safety_df.rename(columns={'title': 'Item Name', 'item_type_keyword': 'Item_type',
                                               'country' : 'Country of Origin', 
                                               'Retail or not Retail': 'Retail or not retail'}, inplace=True)
        


        test_product_safety_df = test_product_safety_df[['Merchant Brand Name', 'Item Name', 'Item_type',
               'Country of Origin', 'Retail or not retail']]
        
        test_product_safety_df['Retail or not retail'].replace(to_replace=['Not Retail'],value='Not_Retail', inplace=True)
        
        print(test_product_safety_df)
        df_with_target_label = test_product_safety_df

        df_with_target_label["Retail or not retail"].fillna(value="Not_Retail", inplace=True)
        
        df_with_target_label = dropRowsWithUnsupportedLanguage(df_with_target_label,'Item Name')


        # Removing special characters is not helping the model being built, optional.
        
        for index, row in df_with_target_label.iterrows():
            print(row['Item Name'])
            row['Item Name'] = removeSpecialCharacters(str(row['Item Name']))
            df_with_target_label.set_value(index, 'Item Name', removeSpecialCharacters(str(row['Item Name'])))
            print(df_with_target_label.get_value(index, 'Item Name', takeable=False))
        
        
        # # Perform stemming and lemmatization on the text columns to get to the root word. 
        #Caveats, if Chinese is in text then stemming wont happen. Snowball stemmer doesnt have python libraries. 
        
        # For this to run you need to install nltk stopwords
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        
        # Stemming
        for index, row in df_with_target_label.iterrows():
            print(row['Item Name'])
            df_with_target_label.set_value(index, 'Item Name', ",".join([ stemmer.stem(kw) for kw in df_with_target_label.get_value(index, 'Item Name', takeable=False).split(" ")]))
            print(df_with_target_label.get_value(index, 'Item Name', takeable=False))
        
        # Do not use str to convert from unicode to encoded text / bytes.
        # As columns are text fill missing values and junk values with 'missing' string.
        df_with_target_label["Merchant Brand Name"].fillna(value="missing", inplace=True)
        df_with_target_label["Item_type"].fillna(value="missing", inplace=True)
        df_with_target_label["Country of Origin"].fillna(value="missing", inplace=True)
        
        for index, row in df_with_target_label.iterrows():
            if(type(row['Item_type']) == float or type(row['Item_type']) == int or type(row['Item_type']) == bool):
                print(row['Item_type'])
                df_with_target_label.set_value(index, 'Item_type', 'missing')
                print(df_with_target_label.get_value(index, 'Item_type', takeable=False))
        
        for index, row in df_with_target_label.iterrows():
            if(type(row['Merchant Brand Name']) == float or type(row['Merchant Brand Name']) == int or type(row['Merchant Brand Name']) == bool):
                print(row['Merchant Brand Name'])
                df_with_target_label.set_value(index, 'Merchant Brand Name', 'missing')
                print(df_with_target_label.get_value(index, 'Merchant Brand Name', takeable=False))
        
        for index, row in df_with_target_label.iterrows():
            if(type(row['Country of Origin']) == float or type(row['Country of Origin']) == int or type(row['Country of Origin']) == bool):
                print(row['Country of Origin'])
                df_with_target_label.set_value(index, 'Country of Origin', 'missing')
                print(df_with_target_label.get_value(index, 'Country of Origin', takeable=False))
        
        indices_to_generate_results = test_product_safety_df.index.values
        print("ATTTRIBUTES")
        print(test_product_safety_df.index.values)
        
        predicted = pipeline.predict(df_with_target_label)
        
        predicted_proba = pipeline.predict_proba(df_with_target_label)
        print(predicted_proba)
        


        export_excel_file = test_product_safety_df.loc[indices_to_generate_results]
        
        
        export_excel_file['PREDICTED'] = pd.Series(predicted, index=export_excel_file.index)
        export_excel_file['prediction-HIGH'] = predicted_proba[:,0]
        export_excel_file['prediction-LOW'] = predicted_proba[:,1]
        export_excel_file['prediction-MEDIUM'] = predicted_proba[:,2]

        buildAsin = []
        for index in indices_to_generate_results:
            buildAsin.append(asin_column[index-1])

        export_excel_file['ASIN'] = buildAsin


        print(export_excel_file)


        
        
        # Save results file
        
        writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')
        
        # Convert the dataframe to an XlsxWriter Excel object.
        export_excel_file.to_excel(writer, sheet_name='Sheet1')

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'bold': True})
        worksheet.write_string(0, 11, 'RISK_LEVEL', format1)
        for row in range(len(predicted)):
            worksheet.write_formula(row + 1, 11, 'IF(G{0}="HIGH", IF(H{0} >= 0.7, 5, 4), IF(G{0} = "MEDIUM", IF(J{0} >= 0.7, 3, 2), 1))'.format(row + 1))

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        self.write('File was saved')



# Create application, set up server for it listening to port 8000, start IO loop
def main():
    application = Application()
    http_server = tornado.httpserver.HTTPServer(application, xheaders=True)
    http_server.listen(9023)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()


'''
# # GridSearchCV for finding best parameters

rf_filter = my_rf_filter(threshold='mean')
clf = RandomForestClassifier(n_jobs=-1, random_state=42, oob_score=False)

# Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.
# Choose lesser number of combination of parameters at a time and experiment. Else it take forever to run.

# Grid search parameters
# rf_n_estimators = [10, 50, 500, 1000]
# rf_max_features = ['auto', 'sqrt', 'log2']
# rf_max_depth = [None, 3, 5, 10, 20]
# rf_min_samples_split = [2, 3, 10]
# rf_min_samples_leaf = [1, 3, 10]
# rf_bootstrap = [True, False]
# rf_criterion = ["gini", "entropy"]

rf_n_estimators = [10, 50, 500, 1000]
rf_max_features = ['auto']
rf_max_depth = [None]
rf_min_samples_split = [2]
rf_min_samples_leaf = [3]
rf_bootstrap = [False]
rf_criterion = ["gini"]

# rff_transform = ["median", "mean"] # Search the threshold parameters

estimator = GridSearchCV(pipeline,
                         cv = 3, 
                         param_grid = dict(clf__n_estimators = rf_n_estimators,
                                          clf__max_features = rf_max_features,
                                          clf__max_depth = rf_max_depth,
                                          clf__min_samples_split = rf_min_samples_split,
                                          clf__min_samples_leaf = rf_min_samples_leaf,
                                          clf__bootstrap = rf_bootstrap,
                                          clf__criterion = rf_criterion))

estimator.get_params().keys()
estimator.fit(X_train, y_train)


# # Running on all ML Algos
classifier_map = {"Decision Tree " : DecisionTreeClassifier(random_state=0),
                   "Bagging Classifier " : BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5),
                   "Random Forest Classifier " : RandomForestClassifier(n_estimators=500, max_depth=None,min_samples_split=2, random_state=42),
                   "AdaBoost" : AdaBoostClassifier(n_estimators=100),
                 "SVC " : SVC(gamma=2, C=1),
                 "Logistic Regression " : linear_model.LogisticRegression(C=1e5)}


for clf_key, clf_value in classifier_map.iteritems():
    pipeline = Pipeline([
    ('union', feature_union),
    ('clf', clf_value),])
    
    pipeline.fit(X_train, y_train)

    predicted = pipeline.predict(X_test)
    score = pipeline.score(X_test,y_test)

    
    print("Accuracy after running with %s algorithm is %f" % (clf_key,score))
    f = open('report_compliance_risk_model.txt','a')
    
    f.write("Accuracy after running with %s algorithm is %f" % (clf_key,score))
    f.write("\n")
    
    print(metrics.classification_report(y_test, predicted,target_names=['LOW','MEDIUM','HIGH']))
    f.write(metrics.classification_report(y_test, predicted,target_names=['LOW','MEDIUM','HIGH']))
    f.write("\n")
    
    print(pd.crosstab(y_test, predicted, rownames=['True'], colnames=['Predicted'], margins=True))
    f.write("\n\n\n")
    
    f.close()

'''