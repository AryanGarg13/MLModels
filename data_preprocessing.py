import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreProcessing:
    def __init__(self):
        pass

    def label_encoding(self,df, col):
        encoded_labels, _ = pd.factorize(df[col])
        df[col] = encoded_labels    
        return df

    def drop_unimportant_cols(self,df,col_list):
        for col in col_list:
            if(col in df.columns):
                df.drop(columns=col,inplace=True)

    def drop_duplicate_genres(self,df):
        df.drop_duplicates(subset= ['track_id'],keep='first',inplace = True)
        return df

    def barplot_feature_distribution(self,df,col,save_path):
        x = (df[col].unique())
        # print(x)
        y = df[col].value_counts().loc[x]
        # print(y)
        # Value_counts returns the frequency of each object in the column .loc gives the value according to x plt.bar automatically sorts x
        plt.bar(x,y)
        plt.title("Bar graph for column " + str(col))
        plt.savefig(save_path)
        plt.close()

    def histplot_feature_distribution(self,df,col,numbins,save_path):
        x = (df[col].unique())
        mini = int(np.floor(min(x)))
        maxi = int(np.ceil(max(x)))
        # print(mini,maxi)
        bins = np.linspace(mini, maxi, numbins+1)
        # print(bins)
        y = df[col]
        # print(y)
        plt.hist(y,bins=bins)
        plt.title("Bar graph for column " + str(col))
        plt.savefig(save_path)
        plt.close()

    def plot_corr_matrix(self,df,save_path):
        corr = df.corr()
        plt.figure(figsize=(20, 20))
        sns.heatmap(corr,annot=True,
            cmap=sns.dark_palette("skyblue", as_cmap=True),
            vmin=-1.0, vmax=1.0,
            square=True)
        self.corr = corr
        plt.savefig(save_path)
        return self.corr
    
    def correlated_columns(self,col_list,threshold = 0.5):
        correlated_cols = set()
        for col in col_list:
            # Get the correlations for the specified column
            high_corr = self.corr[col][abs(self.corr[col]) > threshold].index.tolist()
            # Add all columns except the current one
            correlated_cols.update([x for x in high_corr if x != col])

        return correlated_cols
    

    def remove_outliers(self,df, columns=None):
        if columns is None:
            columns = df.select_dtypes(include=[float, int]).columns

        df_clean = df.copy()

        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        return df_clean

    def plot_boxplots(self,df, columns=None, save_path=None):
        if columns is None:
            columns = df.select_dtypes(include=[float, int]).columns

        # Get the cleaned data
        df_clean = self.remove_outliers(df, columns)

        # Plotting
        plt.figure(figsize=(14, 6))

        # Original data boxplot
        plt.subplot(1, 2, 1)
        df[columns].boxplot()
        plt.title(f'Original Data with Outliers for {columns}')

        # Cleaned data boxplot
        plt.subplot(1, 2, 2)
        df_clean[columns].boxplot()
        plt.title(f'Cleaned Data without Outliers for {columns}')

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path)

        # Close the figure to free up memory
        plt.close()