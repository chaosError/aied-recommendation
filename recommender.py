import streamlit as st
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt 
import altair as alt

@st.cache(allow_output_mutation=True)
def load_data():
	source_path1 = os.path.join("data/coursera-courses-overview.csv")
	source_path2 = os.path.join("data/coursera-individual-courses.csv")
	source_path3 = os.path.join("data/coursera-courses.csv")
	df_overview = pd.read_csv(source_path1)
	df_individual = pd.read_csv(source_path2)
	df_style = pd.read_csv(source_path3)
	df = pd.concat([df_overview, df_individual], axis=1)
	df["learn_style"] = df_style["learn_style"]
	df = prepare_data(df)
	return df

@st.cache(persist=True)
def filter(dataframe, chosen_options, feature, id):
	selected_records = []
	for i in range(1000):
		for op in chosen_options:
			if op in dataframe[feature][i]:
				selected_records.append(dataframe[id][i])
	return selected_records

@st.cache(persist=True)
def prepare_data(df):
	df.columns = clean_col_names(df, df.columns)

	df['skills'] = df['skills'].fillna('Missing')
	df['instructors'] = df['instructors'].fillna('Missing')

	def make_numeric(x):
		if(x=='Missing'):
			return np.nan
		return float(x)

	df['course_rating'] = df['course_rating'].apply(make_numeric)
	df['course_rated_by'] = df['course_rated_by'].apply(make_numeric)
	df['percentage_of_new_career_starts'] = df['percentage_of_new_career_starts'].apply(make_numeric)
	df['percentage_of_pay_increase_or_promotion'] = df['percentage_of_pay_increase_or_promotion'].apply(make_numeric)

	def make_count_numeric(x):
	    if('k' in x):
	        return (float(x.replace('k','')) * 1000)
	    elif('m' in x):
	        return (float(x.replace('m','')) * 1000000)
	    elif('Missing' in x):
	        return (np.nan)

	df['enrolled_student_count'] = df['enrolled_student_count'].apply(make_count_numeric)

	def find_time(x):
	    l = x.split(' ')
	    idx = 0
	    for i in range(len(l)):
	        if(l[i].isdigit()):
	            idx = i 
	    try:
	        return (l[idx] + ' ' + l[idx+1])
	    except:
	        return l[idx]

	df['estimated_time_to_complete'] = df['estimated_time_to_complete'].apply(find_time)

	def split_it(x):
		return (x.split(','))
	df['skills'] = df['skills'].apply(split_it)

	return df

@st.cache(persist=True)
def clean_col_names(df, columns):
	new = []
	for c in columns:
		new.append(c.lower().replace(' ','_'))
	return new


from rake_nltk import Rake
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def extract_keywords(df, feature):
    r = Rake()
    keyword_lists = []
    for i in range(1000):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)
        
    return keyword_lists

def extract_keywords(df, feature):
    r = Rake()
    keyword_lists = []
    for i in range(df[feature].shape[0]):
        descr = df[feature][i]
        r.extract_keywords_from_text(descr)
        key_words_dict_scores = r.get_word_degrees()
        keywords_string = " ".join(list(key_words_dict_scores.keys()))
        keyword_lists.append(keywords_string)
        
    return keyword_lists

def recommendations(df, input_course, cosine_sim, find_similar=True, how_many=5):
    recommended = []
    selected_course = df[df['course_name']==input_course]
    
    idx = selected_course.index[0]

    if(find_similar):
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    else:
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = True)

    if(len(score_series) < how_many):
    	how_many = len(score_series)
    top_sugg = list(score_series.iloc[1:how_many+1].index)
    
    for i in top_sugg:
        qualified = df['course_name'].iloc[i]
        recommended.append(qualified)
        
    return recommended

def style_based_recommendations(df, input_course, courses):

	df = df[df['course_name'].isin(courses)].reset_index()

	df['descr_keywords'] = extract_keywords(df, 'description')

	count = CountVectorizer()
	count_matrix = count.fit_transform(df['descr_keywords'])

	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	rec_courses_similar = recommendations(df, input_course, cosine_sim, True)
	temp_sim = df[df['course_name'].isin(rec_courses_similar)]
	rec_courses_dissimilar = recommendations(df, input_course, cosine_sim, False)
	temp_dissim = df[df['course_name'].isin(rec_courses_dissimilar)]

	st.write("Top 5 最接近的课程推荐")
	st.write(temp_sim)
	st.write("Top 5 最不相似的课程推荐")
	st.write(temp_dissim)

def prep_for_cbr(df):

	st.header("基于学习风格的推荐")
	st.sidebar.header("选项过滤")
	st.write("分析基于**技能**的过滤后的课程子集")
	st.write("根据基于风格的推荐，找到与所选课程相似的课程"
		" 学习者可以选择任何根据自身的技能筛选出来的课程")
	st.write("从侧边栏的'选择课程'下拉菜单中选择课程")

	skills_avail = []
	for i in range(1000):
		skills_avail = skills_avail + df['skills'][i]
	skills_avail = list(set(skills_avail))
	skills_select = st.sidebar.multiselect("选择课程", skills_avail)

	skill_filtered = None
	courses = None
	input_course = "Nothing"

	temp_skills = filter(df, skills_select, 'skills', 'course_url')

	skill_filtered = df[df['course_url'].isin(temp_skills)].reset_index()

	courses = skill_filtered['course_name']
	st.write("### 特征选项过滤")
	st.write(skill_filtered)
	st.write("**筛选出的项目数量:**",skill_filtered.shape[0])
	st.write("**课程数量:**",
		skill_filtered[skill_filtered['learning_product_type']=='COURSE'].shape[0])
	st.write("**专业学位的数量:**",
		skill_filtered[skill_filtered['learning_product_type']=='PROFESSIONAL CERTIFICATE'].shape[0])
	st.write("**专业数量:**",
		skill_filtered[skill_filtered['learning_product_type']=='SPECIALIZATION'].shape[0])

	chart = alt.Chart(skill_filtered).mark_bar().encode(
		y = '被提供的课程:N',
		x = '被提供的课程数:Q'
	).properties(
		title = '提供课程的组织'
	)
	st.altair_chart(chart)

	if(len(courses)<=2):
		st.write("*不得超过2项*")

	input_course = st.sidebar.selectbox("课程设置", courses, key = 'courses')

	rec_radio = st.sidebar.radio("推荐相似课程", ('否', '是'), index = 0)
	if (rec_radio=='是'):
		style_based_recommendations(df, input_course, courses)

def main():

	st.title("基于学习风格的推荐")
	st.write("课程探索")
	st.sidebar.title("参数选择")
	st.sidebar.header("初步检查")
	st.header("项目相关")
	st.write("Style-based Recommendation是一个简单的推荐系统，"
			"旨在帮助学习者由一个数据驱动的策略"
			"在Coursera上的课程中进行导航"
			"学习者可以直观地看到数据集中提供的不同功能，"
			"或者与这个应用程序进行互动"
			"来寻找合适的课程。")

	df = load_data()
	st.header("数据使用")
	st.write("基于构建本系统的目的, 选用Coursera的数据"
		" 最终的数据集由1000个实例和15个特征组成"
		" 我们添加了至关重要的学习风格特征，帮助更好的推荐相似风格视频")
	st.markdown("拨动 **显示原始数据** 侧边栏上的复选框"
		" 来显示隐藏的数据.")

	if st.sidebar.checkbox("显示原始数据", key='显示数据'):
		st.write(df)
	else:
		pass

	st.markdown("### 每个特征代表什么？")
	st.write("**course_url:** 课程主页URL")
	st.write("**course_name:** 课程名")
	st.write("**learning_product_type:** 是课程、专业证书还是专业课")
	st.write("**course_provided_by:** 课程提供合作方")
	st.write("**course_rating:** 对课程的总体评价")
	st.write("**course_rated_by:** 总体评价基于的学生人数")
	st.write("**enrolled_student_count:** 参与的学习者数量")
	st.write("**course_difficulty:** 课程难度")
	st.write("**skills:** 课程关联的技能")
	st.write("**description:** 课程描述")
	st.write("**percentage_of_new_career_starts:**结课后开启新职业道路的学习者人数")
	st.write("**percentage_of_pay_increase_or_promotion:** 结课后升职加薪的人数")
	st.write("**estimated_time_to_complete:** 大约完成的时间")
	st.write("**instructors:** 授课教师")
	st.write("**learn_style:** 最后一个是学习风格，是本系统的重要特征，代表课程视频的大致教学风格，一共有四个类别：A:行动型、B:发散型、C:思考型、D:聚焦型")

	prep_for_cbr(df)
	
	
if __name__=="__main__":
	main()

# Transform following csv scheme feteched from xapi source
#   `course_id,user,recommend1,recommend2,recommend3,recommend4,recommend5,recommend-1,recommend-2,recommend-3,recommend-4,recommend-5`
# into
#   `Course URL,Course Name,Learning Product Type,Course Provided By,Course Rating,Course Rated By,Enrolled Student Count,Course Difficulty,View Time,Similiar Redirect tTime,Dissimilar Redirect Time`
# A small sample csv file in the transformed format for model experimenting has been put in the xapi_data.csv.