-- Databricks notebook source
-- MAGIC %md
-- MAGIC - SQL与DBFS/database/delta table的数据交互
-- MAGIC - pyspark执行sql语句命令
-- MAGIC - sql基础操作。参考教程：https://www.runoob.com/sql/sql-tutorial.html
-- MAGIC   - select {col} from {table}
-- MAGIC   - select distinct {col} from {table}
-- MAGIC   - select {col} from {table} where {col logic value}
-- MAGIC   - select {col} from {table} where {logic} and {logic}
-- MAGIC   - select {col} from {table} order by {col} asc|desc
-- MAGIC   - insert into {table} (col..) values (value..)
-- MAGIC   - update {table} set {col=value..} where {value=xx}
-- MAGIC   - delete from {table} where {value=xx}
-- MAGIC   - select top {number|percent} {col} from {table}
-- MAGIC   - select {col} from {table} where {col} like {pattern - % _ }
-- MAGIC   - select {col} from {table} where {col} in {value1,value2...}
-- MAGIC   - select {col} from {table} where {col} between {value1} and {value2}
-- MAGIC   - select {col} as {new_col} from {table}
-- MAGIC   - select {col1,col2} from {table1} inner join {table2} on {table1}{index1}={table2}{index2}
-- MAGIC   - create table {new_table} as select * from {old_table}
-- MAGIC   - create table {table_name}(col1 datatype1, col2 datatype2...)
-- MAGIC   - create index {index_name} on {table}{col}
-- MAGIC   - drop index {index_name} on {table}
-- MAGIC   - select {col} from {table} where {col} is null
-- MAGIC   - select avg(col) from {table}
-- MAGIC   - select count(col) from {table}
-- MAGIC   - select first(col) from {table}
-- MAGIC   - select last(col) from {table}
-- MAGIC   - select max(col) from {table}
-- MAGIC   - select min(col) from {table}
-- MAGIC   - select sum(col) from {table}
-- MAGIC   - select {col}, avg(col) from {table} where {col operator value} group by {col}
-- MAGIC   - **********************
-- MAGIC   - SELECT column_name, aggregate_function(column_name) 
-- MAGIC   - FROM table_name
-- MAGIC   - WHERE column_name operator value
-- MAGIC   - GROUP BY column_name
-- MAGIC   - HAVING aggregate_function(column_name) operator value
-- MAGIC   - 在 SQL 中增加 HAVING 子句原因是，WHERE 关键字无法与聚合函数一起使用
-- MAGIC   - **********************
-- MAGIC   - EXISTS 运算符用于判断查询子句是否有记录，如果有一条或多条记录存在返回 True，否则返回 False
-- MAGIC   - SELECT column_name(s)
-- MAGIC   - FROM table_name
-- MAGIC   - WHERE EXISTS
-- MAGIC   - (SELECT column_name FROM table_name WHERE condition);
-- MAGIC   - **********************
-- MAGIC   - SELECT LEN(column_name) FROM table_name;
-- MAGIC   - SELECT NOW() FROM table_name;
-- MAGIC - sql开窗函数

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ![alt](https://www.runoob.com/wp-content/uploads/2019/01/sql-join.png)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # 将DBFS中的文件读取为spark dataframe，将dataframe写为delta表
-- MAGIC dftemp = spark.read.csv("/FileStore/tables/train.csv", header="true", inferSchema="true")
-- MAGIC dftemp.write.format("delta").mode("overwrite").save("/delta/dftemp")
-- MAGIC 
-- MAGIC print(dftemp.dtypes)
-- MAGIC 
-- MAGIC dftemp.select('PassengerID','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked').show()

-- COMMAND ----------

-- 通过delta表，将数据写入sql database-default
DROP TABLE IF EXISTS dftemp;
CREATE TABLE dftemp USING DELTA LOCATION '/delta/dftemp'

-- COMMAND ----------

-- 读取表
SELECT * FROM dftemp

-- COMMAND ----------

SELECT
  Pclass,
  avg(Age) AS Age
FROM
  dftemp
GROUP BY
  Pclass
ORDER BY
  Pclass

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # 通过pyspark command运行sql命令
-- MAGIC from pyspark.sql.functions import avg
-- MAGIC 
-- MAGIC display(dftemp.select("Age","Pclass").groupBy("Pclass").agg(avg("Age")).sort("Pclass"))

-- COMMAND ----------

-- 展示全表数据内容
select * from dftemp

-- COMMAND ----------

-- 显示数据库和数据库中的所有表名
-- show databases
show tables

-- COMMAND ----------

-- 条件过滤筛选
select Name, Sex, Age 
from dftemp
where Age>25
and Sex='female'

-- COMMAND ----------

-- 统计表的总行数
select count(*) from dftemp

-- COMMAND ----------

-- 统计去重后的变量数
select distinct Pclass from dftemp

-- COMMAND ----------

-- 显示前多少行的数据
select * from dftemp limit 100

-- COMMAND ----------

-- 按某一列排序(默认升序，如果要降序在order by后面加desc)
select * from dftemp
order by age desc

-- COMMAND ----------

-- 条件筛选，where not <condition>
select Name,Age,Pclass from dftemp 
where Pclass in ('1','2') 
order by Age desc; 


-- COMMAND ----------

-- <通配符 %>条件筛选。 keyword% - 返回以keyword开头的查询结果， %keyword% - 返回包含keyword的结果，A%B - 返回以A开头并以B结尾的结果。
-- 通配符也可以用下划线来替代 _, 只是每一个下划线只能代表一个字符。多个字符需要加多个下划线
select Name from dftemp 
where Name like '%George%'

-- COMMAND ----------

-- 拼接形成新字段
select Name || '-' || Age from dftemp

-- COMMAND ----------

-- 自定义度量值
-- select <col.name1>,<col.name2>,<col.name1>*<col.name2> as <new.col.name> from <table.name>
select Age,Pclass,Age*Pclass as new_col_test from dftemp

-- COMMAND ----------

-- 字符串大小写变更
-- select upper(Name) as new_col_test from dftemp
select lower(Name) as new_col_test from dftemp

-- COMMAND ----------

-- 分组统计结果
select Pclass,count(*) as ClassSum
from dftemp 
group by Pclass


-- select <col.name>,count(*) as <new.col.name> 
-- from <table.name> group by <col.name> having count(*)>= xx

-- query format: select - from - where - group by - having - order by

-- COMMAND ----------

-- <子查询 - 作为where条件>
-- 基于表1的查询结果，作为输入条件应用在表2的查询中，构成子查询(子查询可以是多重，不一定只有1个子查询)

-- select <col.name> from <table.name> 
-- where <col.name1> in (select <col.name2> 
-- from <table.name1> where <col.name3>='xx 

-- COMMAND ----------

-- <子查询2 - 作为select条件,示例>
-- 将子查询作为select的条件加入查询中

-- select cust_name,cust_state,
-- 	(select count(*) from orders 
-- where orders.cust_id = customer.cust_id) as orders 
-- from customers 
-- order by cust_name; 

-- COMMAND ----------

-- <联结表 - join，多表拼接>
-- 为什么要设计多表? 为了更有效的存储，方便地处理；即更好的可伸缩性(scale well)-能够适应不断增加的数据量而不失败. col.name 1,2,3是来自不同表的join字段，不能来自于同一张表；where条件中可以不是primary key,但要注意字段中的重复情况，重复会进行排列组合，让数据加倍。没有where的条件时，返回的结果为笛卡尔积；

-- select <col.name1>,<col.name2>,<col.name3> 
-- from <table.name1>,<table.name2> 
-- where <table.name1>.<col.name1> = <table.name2>.<col.name2>;

-- COMMAND ----------

-- <内联结 - inner join,也叫等值联结equijoin>,示例：

-- select vend_name,prod_name,prod_price
-- from vendors
-- where join products on vendors.vend_id = products.vend_id;
-- <联结多个表>,示例:

-- select prod_name,vend_name,prod_price,quantity
-- from orderitems,products,vendors
-- where products.vend_id = vendors.vend_id
-- 	and orderitems.prod_id = products.prod_id
-- 	and order_num = 20007;
-- <子查询优化 -> 联结多表>,示例:

-- select cust_name,cust_contact
-- from customers
-- where cust_id in (select cust_id
--                  form orders
--                  where order_num in (select order_num
--                                      from orderitems
--                                      where prod_id = 'RGAN01';
--                  					)                 
--                  );
-- 优化后↓
-- select cust_name,cust_contact
-- from customers,orders,orderitems
-- where customers.cust_id = orders.cust_id
-- 	and orderitems.order_num = orders.order_num
-- 	and prod_id = 'RGAN01';
-- <高级联结 - 表别名(优势:缩短语句，多次引用)>,示例：

-- select cust_name, cust_contact
-- from customers as c,orders as o,orderitems as oi
-- where c.cust_id = o.cust_id
-- 	and oi.order_num = o.order_num
-- 	and prod_id = 'RGAN01';
-- <高级联结 - 自联结self-join>:

-- select cust_id,cust_name,cust_contact
-- from customers
-- where cust_name = (select cust_name
--                   from customers
--                   where cust_contact = 'Jim Jones;
-- 优化后↓(性能更佳，自联结通常作为外部语句，用来替代从相同表中检索数据而使用的子查询语句)

-- select c1.cust_id,c1.cust_name,c1.cust_contact
-- from customers as c1,customers as c2
-- where c1.cust_name = c2.cust_name
-- 	and c2.cust_contact = 'Jim Jones';
-- <高级联结 - 自然联结natural-join>:
-- (通配符对第一张表使用，所有其他列明确列出，所有没有重复的列被检索出来)

-- select c.*, o.order_num, o.order_date, oi.prod_id, oi.quantity, oi.item_price
-- from customers as c, orders as o, orderitems as oi
-- where c.cust_id = o.cust_id
-- 	and oi.order_num = o.order_num
-- 	and prod_id = 'RGAN01';
-- <高级联结 - 外联结outer-join>: #(统计“被关联+目标表没被关联到”的数据信息.使用外联结时，必须指定包括其所有行的表(排除联结上的行之外的所有行)，right-指outer join右边的表，left-指outer join左边的表. sqlite中不支持right outer，可以通过调整from-where表的先后顺序来实现。)
-- (内联结)

-- select customers.cust_id,orders.order_num
-- from customers
-- inner join orders on customers.cust_id = orders.cust_id;
-- (外联结)

-- select customers.cust_id, orders.order_num
-- from customers
-- left outer join orders on customers.cust_id = orders.cust_id;
-- <高级联结 - 带聚集函数的联结>,示例-检索所有顾客及每个顾客所下的订单数
-- (方法1)

-- select customers.cust_id,count(orders.order_num) as num_ord
-- from customers
-- inner join orders on customers.cust_id = orders.cust_id
-- group by customers.cust_id
-- (方法2)

-- select customers.cust id,count(orders.order_num) as num_ord
-- from customers
-- left outer join orders on customers.cust_id = orders.cust_id
-- group by customers.cust_id;
-- <组合查询 - union> - 多条select语句合并查询/union中各条select每列都为相同字段

-- select cust_name, cust_contact, cust_email
-- from customers
-- where cust_state in ('IL','IN','MI
-- union
-- where cust_name, cust_contact, cust_email
-- from customers
-- where cust_name = 'Fun4All'
-- <union默认将重复的行会被自动取消，如果不需要取消，用union all>

-- select cust_name, cust_contact, cust_email
-- from customers
-- where cust_state in ('IL','IN','MI
-- union all
-- where cust_name, cust_contact, cust_email
-- from customers
-- where cust_name = 'Fun4All'
-- <union中的order by只能作用于最后一条select语句之后>

-- select cust_name, cust_contact, cust_email
-- from customers
-- where cust_state in ('IL','IN','MI
-- union
-- where cust_name, cust_contact, cust_email
-- from customers
-- where cust_name = 'Fun4All'
-- order by cust_name, cust_contact
-- <插入数据 - insert> - 插入完整的行,每个值将按默认顺序插入

-- insert into customers
-- value('123','abc','thooo;
-- <insert - 更保险的方式,分别给定列名和值，一一对应>

-- insert into customers(cust_id,cust_contact,cust_name)
-- value('123','abc','thooo
-- <insert 检索后的结果数据 - customers.new与customer表结构相同，但主键cust_id不能相同>

-- insert into customers(cust_id,cust_contact,cust_name)
-- select cust_id,cust_contact,cust_name
-- from customers.new
-- <从一个表复制到另一个表>

-- create table table.new as select * from old.table;
-- <更新和删除数据 - insert/delete> - 注意跟where条件，否则更新全表

-- update customers
-- set cust_email = 'newemail@xxx.com'
-- where cust_id = '1000000';
-- <更新多列>

-- update customers
-- set cust_email = 'newemail@xxx.com',cust_contact='bob'
-- where cust_id = '1000000';
-- <删除某列的值> - 删除cust_email列

-- update customers
-- set cust_email=null
-- where cust_id='1000000';
-- <删除行>

-- delect customers
-- where cust_id='100000';
-- <创建表> - 表名，表定义

-- create table product
-- (
-- 	prod_id	char(10) not null,
--     vend_id char(10) not null,
--     prod_name decimal(8,2) not null,
-- )
-- <更新表 - alert> - 增加一列

-- alter table vendors
-- add vend_phone char(20);
-- <alert - 删除一列>

-- alter table vendors
-- drop column vend_phone;
-- <删除表>

-- drop table custcopy;
-- <视图-view|虚拟表> 作用：重用sql语句，简化复杂sql操作，使用表的一部分而不是整体，权限控制和分享等 利用视图简化复杂的联结

-- create view productcustomers as 
-- select cust_name,cust_contact,prod_id
-- from customers,orders,orderitems
-- where customers.cust_id=orders.cust_id
-- 	and orderitems.order_num=orders.order_num;
	
-- select cust_name,cust_contact
-- from productcustomers
-- where prod_id ='RGAN01';
-- <使用视图创建常用的数据格式>

-- create view vendorlocation as
-- select rtrim(vend_name) + '(' + RTRIM(vend_country)+'
-- 	as vend_title
-- from vendors;
-- <使用视图过滤不想要的数据>

-- create view customeremailist as
-- select cust_id, cust_name, cust_email
-- from customers
-- where cust_email is not null;
-- <使用视图检索计算结果>

-- create view orderitemexpanded as
-- select order_num, prod_id, quantity, item_price, quantity*item_price as expanded_price
-- from orderitems

-- select * 
-- from orderitemexpanded
-- where order_num=20008;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC SQL开窗函数，函数名(列)+over()。 
-- MAGIC - 聚合开窗函数
-- MAGIC - 排序开窗函数
-- MAGIC - 其他开窗函数
-- MAGIC 
-- MAGIC 开窗函数参考实例：https://zhuanlan.zhihu.com/p/450298503

-- COMMAND ----------

-- sql开窗函数，相当于增强版的聚合函数，起到聚合函数和子查询的目的，且语法更简洁。开窗函数对一组值进行操作，它不像普通聚合函数那样需要使用GROUP BY子句对数据进行分组，能够在同一行中同时返回基础行的列和聚合列。

-- 开窗函数的语法形式为：函数名(列) + over(partition by <分组用列> order by <排序用列>)，表示对数据集按照分组用列进行分区，并且并且对每个分区按照函数聚合计算，最终将计算结果按照排序用列排序后返回到该行。括号中的两个关键词partition by 和 order by 可以只出现一个。注意：开窗函数不会互相干扰，因此在同一个查询语句中可以同时使用多个开窗函数

-- 聚合开窗函数 / 聚合函数(列) OVER (选项)，这里的选项可以是PARTITION BY子句
-- 排序开窗函数 / 排序函数(列) OVER(选项)，这里的选项可以是ORDER BY子句

-- order by 默认统计范围是 rows between unbounded preceding and current row，也就是取当前行数据与当前行之前的数据运算