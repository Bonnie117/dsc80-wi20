
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    
    from re import search
    dic = {}
    key_list = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    rest_column_names = []
    for j in grades.columns:
        if '-' in j:
            rest_column_names.append(j)
    new_colum = grades.copy()
    new_colum = new_colum.drop(columns = rest_column_names)
    for i in key_list:
        dic[i] = []
        for j in new_colum.columns:
            if i in j or i.capitalize() in j:
                if search(i+'[0-9]{2}$',j) :
                    dic[i].append(j)
                elif search(i.capitalize(),j):
                    dic[i].append(j)
                elif search(i+'[a-zA-Z]+[0-9]{2}$',j) :
                    dic[i].append(j)
    return dic


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    
    student_score_list = []
    student_score_max = []
    student_score = pd.Series()
    for col in grades.columns:
        if 'project' in col and not 'Max' in col and not 'Lateness' in col:
            student_score_list.append(col)
        if 'project' in col and 'Max' in col and not 'Lateness' in col:
             student_score_max.append(col)
    student_score_max = grades[student_score_max].sum(axis = 1)
    student_score_sum = grades[student_score_list].sum(axis = 1)/student_score_max[1]
    
    return student_score_sum




# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    
    student_lab_lateness = []
    for col in grades.columns:
        if 'lab' in col and not 'Max' in col and 'Lateness' in col:
            student_lab_lateness.append(col)
    student_lab = grades[student_lab_lateness]
    for col in student_lab:
        student_lab[col] = student_lab[col].astype(str).str.replace(':','').astype(float)
    for col in student_lab:
        student_lab[col] = student_lab[col].between(1,60000)
    new_name = []
    for i in student_lab.columns:
        new_name.append(i[0:5])
    student_lab.columns = new_name
    return student_lab.sum()


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------
    
def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.8, 0.5}
    True
    """
        
    col = col.apply(lambda x: float(x.replace(':','')))
    def late_val(x):
        if x < 50000:
            return 1.0
        elif x < 1680000 and x > 30000:
            return 0.9
        elif x >= 168000 and x < 3360000:
            return 0.8
        else:
            return 0.5
    col = col.apply(late_val)
    return col


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """

    new = grades.copy()
    new = new.fillna(0)
    names = get_assignment_names(grades)['lab']
    for i in names:
        new[i] = (new[i]/new[i+' - Max Points'])*lateness_penalty(new[i+' - Lateness (H:M:S)'])
    return new[names]


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """

    processed = processed.fillna(0)
    result = (processed.sum(axis = 1)-processed.min(axis = 1))/(len(processed.columns)-1)
    return result

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
        
    labs = 0.2
    projects = 0.3
    checkpointp = 0.025
    disc = 0.025
    midterm = 0.15 
    final = 0.30
    new = grades.copy()
    new = new.fillna(0)
    project_score = projects_total(grades)
    Lab_score =lab_total(process_labs(grades))
    checkpoints = get_assignment_names(grades)['checkpoint']
    discussions = get_assignment_names(grades)['disc']
    for i in checkpoints:
        new[i] = new[i]/new[i+' - Max Points']
    check_score = new[checkpoints].mean(axis = 1)

    for i in discussions:
        new[i] = new[i]/new[i+' - Max Points']
    disc_score = new[discussions].mean(axis = 1)
    
    midterm_scores = new['Midterm']/new['Midterm - Max Points']
    final_exam_scores = new['Final']/new['Final - Max Points']
    final_scores = Lab_score * labs + project_score * projects + check_score * checkpointp + disc_score *disc + midterm *midterm_scores + final_exam_scores * final
    return final_scores

def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """

    def letter(x):
        if x>=0.9:
            return 'A'
        elif x<0.9 and x>=0.8:
            return 'B'
        elif x<0.8 and x>=0.7:
            return 'C'
        elif x<0.7 and x>=0.6:
            return 'D'
        else:
            return 'F'
    return pd.Series(list(map(letter, total.tolist())))


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    dic = {}
    letter_series = final_grades(total_points(grades))
    return letter_series.value_counts()/float(len(grades))

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of sophomores
    was no better on average than the class
    as a whole (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """

    grades = grades.fillna(0.0)
    num_soph = grades[grades['Level'] == 'SO'].shape[0]
    class_size = grades.shape[0]
    soph = grades[grades['Level'] == 'SO']
    soph_grade = total_points(soph).mean()
    other_grade = total_points(grades)
    result = []
    for i in range(N):
        sample = np.random.choice(other_grade, replace=False, size=num_soph)
        result.append(np.mean(sample))
    return (pd.Series(result) >= soph_grade).mean()


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    labs = 0.2
    projects = 0.3
    checkpointp = 0.025
    DI = 0.025
    mt = 0.15 
    final = 0.30
    data1 = grades.copy()
    data = data1.fillna(0)
    
    lab_names = get_assignment_names(grades)['lab']
    labs_ori = process_labs(data)
    for i in labs_ori.columns:
        data[i] = labs_ori[i]
    
    proj_names = get_assignment_names(grades)['project']
    dic = {}
    for i in proj_names:
        try:
            raw = data[i] + data[i+ '_free_response']
            maxi = data[i + ' - Max Points'] + data[i+ '_free_response - Max Points']
            data[i] = (raw/maxi) 
        except:   
            raw = data[i] 
            maxi = data[i + ' - Max Points']
            data[i] = (raw/maxi)
            
    
    
    checkpoints = get_assignment_names(grades)['checkpoint']
    discussions = get_assignment_names(grades)['disc']
    
    for i in checkpoints:
        data[i] = data[i]/data[i+' - Max Points']


    for i in discussions:
        data[i] = data[i]/data[i+' - Max Points']

    
    midterm_scores = data['Midterm']/data['Midterm - Max Points']
    final_exam_scores = data['Final']/data['Final - Max Points']
    data['Midterm'] = midterm_scores
    data['Final'] = final_exam_scores
    
    selected_col = lab_names + proj_names + checkpoints + discussions + ['Midterm'] +['Final']
    selected_data = data[selected_col]
    
    manipulate = selected_data + np.random.normal(0, 0.02, size=(selected_data.shape[0], selected_data.shape[1]))
    for i in manipulate.columns:
        manipulate[i] = np.clip(manipulate[i],0,1)
        
    lab_new = lab_total(manipulate[lab_names]) 
    proj_new = manipulate[proj_names].mean(axis = 1)
    check_new = manipulate[checkpoints].mean(axis = 1)
    disc_new = manipulate[discussions].mean(axis = 1)
    midterm_new = manipulate['Midterm']
    final_new = manipulate['Final']
    
    final_scores = lab_new * labs + proj_new * projects + check_new * checkpointp + disc_new *DI + mt *midterm_new+ final_new * final
    #return final_scores.mean() == total_points(grades).mean()
    return final_scores

# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4], bool)
    True
    """

    return [0.00069610, 82.878, [79.794392, 85.93925], 0.068897, True]

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
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
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
