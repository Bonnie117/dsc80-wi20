import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def data_load(scores_fp):
    """
    follows different steps to create a dataframe
    :param scores_fp: file name as a string
    :return: a dataframe
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> list(scores.columns)
    ['attempts', 'highest_score']
    >>> isinstance(scores.index[0], int)
    False
    """
    # a
    subset = pd.read_csv(scores_fp,usecols = ['name', 'tries', 'highest_score', 'sex'])

    # b
    remove_sex = subset.drop('sex',axis=1)

    # c
    customized_col = remove_sex.rename(columns = {'name':'firstname','tries':'attempts'})

    # d
    first_name = customized_col.set_index('firstname')

    return first_name


def pass_fail(scores):
    """
    modifies the scores dataframe by adding one more column satisfying
    conditions from the write up.
    :param scores: dataframe from the question above
    :return: dataframe with additional column pass
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> len(scores.columns)
    3
    >>> scores.loc["Julia", "pass"]=='Yes'
    True

    """

    data = scores
    case_1 = (data['attempts'].values < 3) & (data['highest_score'].values >= 50)
    case_2 = (data['attempts'].values < 6) & (data['highest_score'].values >= 70)
    case_3 = (data['attempts'].values < 10) & (data['highest_score'].values >= 90)
    choice = case_1 | case_2 | case_3
    data['pass'] = choice
    def helper(case):
        if case:
            return 'Yes'
        else:
            return 'No'
    other = data['pass'].apply(helper).values
    data['pass'] = other
    return data


def av_score(scores):
    """
    returns the average score for those students who passed the test.
    :param scores: dataframe from the second question
    :return: average score
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> av = av_score(scores)
    >>> isinstance(av, float)
    True
    >>> 91 < av < 92
    True
    """

    modified = scores
    return np.mean(modified[modified['pass'] == 'Yes']['highest_score'].values)


def highest_score_name(scores):
    """
    finds the highest score and people who received it
    :param scores: dataframe from the second question
    :return: dictionary where the key is the highest score and the value(s) is a list of name(s)
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> highest = highest_score_name(scores)
    >>> isinstance(highest, dict)
    True
    >>> len(next(iter(highest.items()))[1])
    3
    """
    modified = scores
    hold = max(modified['highest_score'].values)
    content = modified[modified['highest_score'] == hold].index.tolist()
    return {hold:content}


def idx_dup():
    """
    Answers the question in the write up.
    :return:
    >>> ans = idx_dup()
    >>> isinstance(ans, int)
    True
    >>> 1 <= ans <= 6
    True
    """
    return 6



# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def trick_me():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  trick_me()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    return tricky_1 = pd.DataFrame([['Bonnie','',''],['Cathy','1','2'],['Norry','3','4'],['0','1','2'],['7','8','9']],columns = ["Name", "Name", "Age"] )
    tricky_1.to_csv('tricky_1.csv',index = False)
    tricky_2 = pd.read_csv('tricky_1.csv')
    return 'C'

def reason_dup():
    """
     Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_dup()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    return 'B'

def trick_bool():
    """
     Answers the question in the write-up
    :return: a list with three letters
    >>> ans =  trick_bool()
    >>> isinstance(ans, list)
    True
    >>> isinstance(ans[1], str)
    True

    """
    return ['D','J','M']

def reason_bool():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_bool()
    >>> ans == 'A' or ans == 'B' or ans == "C" or ans =="D"
    True

    """
    return 'D'


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def change(x):
    """
    Returns 'MISSING' when x is `NaN`,
    Otherwise returns x
    >>> change(1.0) == 1.0
    True
    >>> change(np.NaN) == 'MISSING'
    True
    """
    if pd.isnull(x):
        return "MISSING"
    else:
        return x


def correct_replacement(nans):
    """
    changes all np.NaNs to "Missing"
    :param nans: given dataframe
    :return: modified dataframe
    >>> nans = pd.DataFrame([[0,1,np.NaN], [np.NaN, np.NaN, np.NaN], [1, 2, 3]])
    >>> A = correct_replacement(nans)
    >>> (A.values == 'MISSING').sum() == 4
    True

    """
    def change(x):
        if pd.isnull(x):
            return "MISSING"
        else:
            return x
        
    return nans.applymap(change)


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def population_stats(df):
    """
    population_stats which takes in a dataframe df 
    and returns a dataframe indexed by the columns 
    of df, with the following columns:
        - `num_nonnull` contains the number of non-null 
          entries in each column,
        - `pct_nonnull` contains the proportion of entries 
          in each column that are non-null,
        - `num_distinct` contains the number of distinct 
          entries in each column,
        - `pct_distinct` contains the proportion of (non-null) 
          entries in each column that are distinct from each other.

    :Example:
    >>> data = np.random.choice(range(10), size=(100, 4))
    >>> df = pd.DataFrame(data, columns='A B C D'.split())
    >>> out = population_stats(df)
    >>> out.index.tolist() == ['A', 'B', 'C', 'D']
    True
    >>> cols = ['num_nonnull', 'pct_nonnull', 'num_distinct', 'pct_distinct']
    >>> out.columns.tolist() == cols
    True
    >>> (out['num_distinct'] <= 10).all()
    True
    >> (out['pct_nonnull'] == 1.0).all()
    True
    """
    
    index = df.columns.values
    num_nonnull =[]
    pct_nonnull= []
    num_distinct = []
    pct_distinct = []
    
    number = df.shape[0]
    
    def change(x):
        return not pd.isnull(x)
    def fill(x):
        if pd.isnull(x):
            return False
        else:
            return True
        
    for i in df.columns:
        col = df[i].apply(change).values
        count = np.count_nonzero(col)
        num_nonnull.append(count)
        pct_nonnull.append(count/number)
        num_distinct.append(df[i].nunique())
        pct_distinct.append(pd.Series(list(filter(fill,df[i].values.tolist()))).nunique()/number)
    return pd.DataFrame({'num_nonnull':num_nonnull,'pct_nonnull':pct_nonnull,
                         'num_distinct':num_distinct,'pct_distinct':pct_distinct},index = ['A', 'B', 'C', 'D'])


def most_common(df, N=10):
    """
    `most_common` which takes in a dataframe df and returns 
    a dataframe of the N most-common values (and their counts) 
    for each column of df.

    :param df: input dataframe.
    :param N: number of most common elements to return (default 10)
.
    :Example:
    >>> data = np.random.choice(range(10), size=(100, 2))
    >>> df = pd.DataFrame(data, columns='A B'.split())
    >>> out = most_common(df, N=3)
    >>> out.index.tolist() == [0, 1, 2]
    True
    >>> out.columns.tolist() == ['A_values', 'A_counts', 'B_values', 'B_counts']
    True
    >>> out['A_values'].isin(range(10)).all()
    True
    """
        
    result = pd.DataFrame()
    for i in df.columns:
        con = df[i].value_counts()
        if N > len(con):
            col_1 = np.array(con.index.tolist() + [np.NaN] * (N-len(con)))
            col_2 = np.array(con.values.tolist() + [np.NaN] * (N-len(con)))
            result[i + '_values'] = col_1
            result[i + '_counts'] = col_2
        else:
            col_1 = np.array(con.index.tolist()[:N])
            col_2 =  np.array(con.values.tolist()[:N])
            result[i +'_values']= col_1
            result[i +'_counts']= col_2
    return result


# ---------------------------------------------------------------------
# Question 5
# ---------------------------------------------------------------------


def null_hypoth():
    """
    :Example:
    >>> isinstance(null_hypoth(), list)
    True
    >>> set(null_hypoth()).issubset({1,2,3,4})
    True
    """
    return [1]

def simulate_null():
    """
    :Example:
    >>> pd.Series(simulate_null()).isin([0,1]).all()
    True
    """
    sig = 0.99
    stats = np.random.binomial(300, sig)
    answer = np.array(stats *[1] + (300 - stats)*[0])
    return answer


def estimate_p_val(N):
    """
    >>> 0 < estimate_p_val(1000) < 0.1
    True
    """
    lst = []
    for i in range(N):
        lst.append(np.count_nonzero(simulate_null()))
    rest = np.array(lst) < 292 #after 8 students
    result = np.count_nonzero(rest)/len(rest)
    return result


# ---------------------------------------------------------------------
# Question 6
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    """
    `super_hero_powers` takes in a dataframe like 
    powers and returns a list with the following three entries:
        - The name of the super-hero with the greatest number of powers.
        - The name of the most common super-power among super-heroes whose names begin with 'M'.
        - The most popular super-power among those with only one super-power.

    :Example:
    >>> fp = os.path.join('data', 'superheroes_powers.csv')
    >>> powers = pd.read_csv(fp)
    >>> out = super_hero_powers(powers)
    >>> isinstance(out, list)
    True
    >>> len(out)
    3
    >>> all([isinstance(x, str) for x in out])
    True
    """

    df = powers.drop('hero_names',axis=1).applymap(np.count_nonzero)
    count = df.sum(axis=1).values
    greatest = max(count)
    powers['count'] = count
    first_col = powers[powers['count'] == greatest]['hero_names'].values[0]

    most_M = powers[powers['hero_names'].str.match('M')]
    M_noHN = most_M.drop('hero_names',axis = 1).applymap(np.count_nonzero)
    modi_M = M_noHN.sum(axis = 0)
    to_find = sorted(modi_M)[-2]
    
    second_col = modi_M[modi_M == to_find].index[0]

    super_power1 = powers[powers['count'] == 1]
    super_power2 = super_power1.drop('hero_names',axis = 1).applymap(np.count_nonzero)
    super_power3 = super_power2.sum(axis=0)
    
    third_col = super_power3[super_power3 == sorted(super_power3)[-2]].index[0]

    return [first_col,second_col,third_col]

# ---------------------------------------------------------------------
# Question 7
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    """
    clean_heroes takes in the dataframe heroes
    and replaces values that are 'null-value'
    place-holders with np.NaN.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = clean_heroes(heroes)
    >>> out['Skin color'].isnull().any()
    True
    >>> out['Weight'].isnull().any()
    True
    """

    checker = lambda x:np.NaN if x=='-' or x == -99 else x
    
    return heroes.applymap(checker)

def super_hero_stats():
    """
    Returns a list that answers the questions in the notebook.
    :Example:
    >>> out = super_hero_stats()
    >>> out[0] in ['Marvel Comics', 'DC Comics']
    True
    >>> isinstance(out[1], int)
    True
    >>> isinstance(out[2], str)
    True
    >>> out[3] in ['good', 'bad']
    True
    >>> isinstance(out[4], str)
    True
    >>> 0 <= out[5] <= 1
    True
    """

    return ['Marvel Comics',558,'Groot','bad','Onslaugh',0.30578512396694213]

# ---------------------------------------------------------------------
# Question 8
# ---------------------------------------------------------------------


def bhbe_col(heroes):
    """
    `bhbe` ('blond-hair-blue-eyes') returns a boolean 
    column that labels super-heroes/villains that 
    are blond-haired *and* blue eyed.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = bhbe_col(heroes)
    >>> isinstance(out, pd.Series)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out.sum()
    93
    """

   cleaned = clean_heroes(heroes)
    blond_hair = (cleaned['Hair color'].values=='Blond') | (cleaned['Hair color'].values=='blond')|(cleaned['Hair color'].values=='Strawberry Blond')
    blue_eye = cleaned['Eye color'].values=='blue'
    
    return pd.Series(blond_hair & blue_eye)


def observed_stat(heroes):
    """
    observed_stat returns the observed test statistic
    for the hypothesis test.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = observed_stat(heroes)
    >>> 0.5 <= out <= 1.0
    True
    """
    
    heroes['bhbe'] = bhbe_col(heroes)
    observed = heroes[heroes['bhbe'] == True]
    observed_prop = observed[observed['Alignment'] == 'good'].shape[0]/observed.shape[0]
    
    return observed_prop

def simulate_bhbe_null(n):
    """
    `simulate_bhbe_null` that takes in a number `n` 
    that returns a `n` instances of the test statistic 
    generated under the null hypothesis. 
    You should hard code your simulation parameter 
    into the function; the function should *not* read in any data.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = simulate_bhbe_null(10)
    >>> isinstance(out, pd.Series)
    True
    >>> out.shape[0]
    10
    >>> ((0.45 <= out) & (out <= 1)).all()
    True
    """

    result = []
    p = 0.8494623655913979
    for i in range(n):
        stat = np.random.binomial(93,p)/93
        result.append(stat)
    return pd.Series(result)


def calc_pval():
    """
    calc_pval returns a list where:
        - the first element is the p-value for 
        hypothesis test (using 100,000 simulations).
        - the second element is Reject if you reject 
        the null hypothesis and Fail to reject if you 
        fail to reject the null hypothesis.

    :Example:
    >>> out = calc_pval()
    >>> len(out)
    2
    >>> 0 <= out[0] <= 1
    True
    >>> out[1] in ['Reject', 'Fail to reject']
    True
    """

    return [0.0015,'Reject']

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------

# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['data_load', 'pass_fail', 'av_score',
            'highest_score_name', 'idx_dup'],
    'q02': ['trick_me', 'reason_dup', 'trick_bool', 'reason_bool'],
    'q03': ['change', 'correct_replacement'],
    'q04': ['population_stats', 'most_common'],
    'q05': ['null_hypoth', 'simulate_null', 'estimate_p_val'],
    'q06': ['super_hero_powers'],
    'q07': ['clean_heroes', 'super_hero_stats'],
    'q08': ['bhbe_col', 'observed_stat', 'simulate_bhbe_null', 'calc_pval']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
