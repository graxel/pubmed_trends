# pubmed_trends

I extract the abstract, title, author, and publication date of all of Pubmed's 36 million science articles. I then filter the articles with a keyword and by date, keeping only articles published in 2005 or later and having to do with cancer.

I fit a BERTopic model to the article abstracts and plot the relative topic frequency over time. The idea is to measure how the "buzz" of different topics in cancer research has evolved over the last 20 years.



### Future Ideas 

Show author activity over time  
    - will need to make a table with (year, author) as primary key and pub_count per year as data

Show topic % vs time  
    - will need to make a table with (year, topic) as primary key and topic_freq per year as data

