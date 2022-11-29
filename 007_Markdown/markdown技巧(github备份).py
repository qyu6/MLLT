# Databricks notebook source
# MAGIC %md
# MAGIC (Markdown笔记)
# MAGIC ---
# MAGIC <常规命令>
# MAGIC - 代码 ` `` `
# MAGIC - 代码块 ` ```xx``` `
# MAGIC - 标题 ` # ## .. `
# MAGIC - 换行 ` </br> `
# MAGIC - 分隔符 ` --- `
# MAGIC - 表格 (第二行 : 表示对其方式。 :---左对齐，:---:居中对其，---:右对齐)
# MAGIC     ```
# MAGIC     | column a | column b |
# MAGIC     | ---: | :--- | 
# MAGIC     | content a | content b |
# MAGIC     | xx | xx |
# MAGIC     ```
# MAGIC - 加粗 ` <b> xx </b> 或者 **xx** `
# MAGIC - 超链接 ` [超链接文本] (超链接url) `
# MAGIC - 图片 ` ![alt] (图片path) `
# MAGIC - 引用 ` > xxx `
# MAGIC - 改变字体颜色 `<font color='red'> xxx </font>`
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ---
# MAGIC <数学公式>
# MAGIC - 行间公式 (自动换行,语法与行内公式都相同↓)`$$ xx $$` $$ xx $$
# MAGIC - 行内公式 $ xx $
# MAGIC     - 估计值 $\hat{x}$  ,`$\hat{x}$`
# MAGIC     - 分数 $\frac{a}{b}$ ,`$\frac{a}{b}$`
# MAGIC     - 累加 $\sum_{a}^{b}$, `$\sum_{a}^{b}$`
# MAGIC     - 累乘 $\prod_{a}^{b}$, `$\prod_{a}^{b}$`
# MAGIC     - 上标 $a^{b}$,`$a^{b}$`
# MAGIC     - 下标 $a_{b}$,`$a_{b}$`
# MAGIC     - 算数平均数 $\overline{x}$ `$\overline{x}$`
# MAGIC     - 底标 $\underset {\Theta} min$ `$\underset {\Theta} min$`
# MAGIC     - $\dot a$ `$\dot a$`
# MAGIC     - $\dots$ `$\dots$`
# MAGIC     - $\vdots$ `$\vdots$`
# MAGIC     - $\ddots$ `$\ddots$`
# MAGIC     - 自适应高度的括号(如包含矩阵的括号) 
# MAGIC     $\left( 
# MAGIC         \begin{bmatrix}
# MAGIC         a\\
# MAGIC         b\\
# MAGIC         c\\
# MAGIC         d
# MAGIC         \end{bmatrix}
# MAGIC     \right)$
# MAGIC     ```
# MAGIC     $\left( 
# MAGIC         \begin{bmatrix}
# MAGIC         a\\
# MAGIC         b\\
# MAGIC         c\\
# MAGIC         d
# MAGIC         \end{bmatrix}
# MAGIC     \right)$
# MAGIC     ```
# MAGIC     - 逆矩阵 $A^{(-1)}$,`$A^{(-1)}$`
# MAGIC     - 矩阵转置 $A^T$,`$A^T$`
# MAGIC     - 单位矩阵 $I$,`$I$`
# MAGIC     - $\sqrt{a+b}$,`$\sqrt{a+b}$`
# MAGIC     - $\sqrt[n]{a+b}$,`$\sqrt[n]{a+b}$`
# MAGIC     - 对数 $ln{a+b}$,`$ln{a+b}$`
# MAGIC     - 对数 $\log_{a}^{b}$,`$\log_{a}^{b}$`
# MAGIC     - 向量 $\vec{a}$,`$\vec{a}$`
# MAGIC     - 方括号 $[,\big[,\bigg[,\Big[,\Bigg[$, `$[,\big[,\bigg[,\Big[,\Bigg[$`
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC     - 花括号（在花括号前加一个斜线，花括号默认在公式内不显示） $\{$ 或 $\}$,`\{,\}`
# MAGIC     - 矩阵表示模板
# MAGIC         $$
# MAGIC         \begin{matrix}
# MAGIC         0&1&1\\
# MAGIC         0&1&1\\
# MAGIC         0&1&1\\
# MAGIC         \end{matrix}
# MAGIC         $$
# MAGIC         ```
# MAGIC         $$
# MAGIC         \begin{matrix}
# MAGIC         0&1&1\\
# MAGIC         0&1&1\\
# MAGIC         0&1&1\\
# MAGIC         \end{matrix}
# MAGIC         $$
# MAGIC         ```
# MAGIC     - 矩阵边框(将模板中matrix替换为如下表示)
# MAGIC         - 小括号边框 `pmatrix`
# MAGIC         - 中括号边框 `bmatrix`
# MAGIC         - 大括号边框 `Bmatrix`
# MAGIC         - 单竖线边框 `vmatrix`
# MAGIC         - 双竖线边框 `Vmatrix`
# MAGIC     - 单边大括号
# MAGIC         $$
# MAGIC         a = \left\{
# MAGIC         \begin{matrix}
# MAGIC         b\\
# MAGIC         c
# MAGIC         \end{matrix}
# MAGIC         \right.
# MAGIC         $$
# MAGIC         ```
# MAGIC         $$
# MAGIC         a = \left\{
# MAGIC         \begin{matrix}
# MAGIC         b\\
# MAGIC         c
# MAGIC         \end{matrix}
# MAGIC         \right.
# MAGIC         $$
# MAGIC         ```
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC     - 特殊数学字符
# MAGIC         - $\mathbb R^{a\times b}$ `$\mathbb R^{a\times b}$`（矩阵符号 ）
# MAGIC         - $\partial$, `\partial` (偏导 | partial derivative )
# MAGIC         - $\alpha$, `$\alpha$`
# MAGIC         - $\beta$, `$\beta$`
# MAGIC         - $\gamma$, `$\gamma$`
# MAGIC         - $\delta$, `$\delta$`
# MAGIC         - $\zeta$, `$\zeta$`
# MAGIC         - $\lambda$, `$\lambda$`
# MAGIC         - $\mu$, `$\mu$`
# MAGIC         - $\nu$, `$\nu$`
# MAGIC         - $\xi$, `$\xi$`
# MAGIC         - $\omicron$, `$\omicron$`
# MAGIC         - $\pi$, `$\pi$`
# MAGIC         - $\rho$, `$\rho$`
# MAGIC         - $\sigma$, `$\sigma$`
# MAGIC         - $\tau$, `$\tau$`
# MAGIC         - $\upsilon$, `$\upsilon$`
# MAGIC         - $\phi$, `$\phi$`
# MAGIC         - $\varphi$, `$\varphi$`
# MAGIC         - $\chi$, `$\chi$`
# MAGIC         - $\psi$, `$\psi$`
# MAGIC         - $\omega$, `$\omega$`
# MAGIC         - $\epsilon$, `$\epsilon$`
# MAGIC         - $\varepsilon$, `$\varepsilon$`
# MAGIC         - $\eta$, `$\eta$`
# MAGIC         - $\theta$, `$\theta$`
# MAGIC         - $\iota$, `$\iota$`
# MAGIC         - $\kappa$, `$\kappa$`
# MAGIC         - $\vert\vert x \vert\vert$,`$\vert\vert x \vert\vert$`, 向量 $x$ 的范式
# MAGIC         - $\in$, `$\in$`
# MAGIC         - $\ni$, `$\ni$`
# MAGIC         - $\notin$, `$\notin$` 
# MAGIC         - $\pm$, `$\pm$`
# MAGIC         - $\times$, `$\times$`
# MAGIC         - $\ast$, `$\ast$`
# MAGIC         - $\mid$, `$\mid$`
# MAGIC         - $\leq$, `$\leq$`
# MAGIC         - $\geq$, `$\geq$`
# MAGIC         - $\ll$, `$\ll$`
# MAGIC         - $\gg$, `$\gg$`
# MAGIC         - $\neq$, `$\neq$`
# MAGIC         - $\approx$, `$\approx$`
# MAGIC         - $\lim$, `$\lim$`
# MAGIC         - $\infty$, `$\infty$`
# MAGIC         - $\uparrow$, `$\uparrow$`
# MAGIC         - $\downarrow$, `$\downarrow$`
# MAGIC         - $\emptyset$, `$\emptyset$` 空集
# MAGIC         - $\subset$ $\supset$, `$\subset$ $\supset$` 子集
# MAGIC         - $\not\subset$, `$\not\subset$` 非子集
# MAGIC         - $\subseteq$ $\supseteq$, `$\subseteq$ $\supseteq$` 真子集
# MAGIC         - $\cup$ $\bigcup$, `$\cup$ $\bigcup$` 并集
# MAGIC         - $\cap$ $\bigcap$, `$\cap$ $\bigcap$` 交集
# MAGIC         - $\prime$, `$\prime$` 导数
# MAGIC         - $\int$, `$\int$` 积分
# MAGIC         - $\iint$, `$\iint$` 双重积分
# MAGIC         - $\iiint$, `$\iiint$` 三重积分
# MAGIC         - $\oint$, `$\oint$` 曲线积分
# MAGIC         - $\nabla$, `$\nabla$` 梯度
# MAGIC         - $\sim$, `$\sim$`
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ---
# MAGIC 特殊问题
# MAGIC - Vscode中如何使用markdown导出pdf
# MAGIC     - 首先在vscode中安装Markdown Pdf插件
# MAGIC     - 初始安装完后，在markdown文件内右键点击会出现导出pdf的选项。但导出pdf后发现公式没有被正确识别渲染，全部是以源码形式导出
# MAGIC     - 找到vscode用户路径：`C://Users/<username>/.vscode/extensions/yzane.markdown-pdf-1.4.1/template/template.html`
# MAGIC     - 打开这个文件，在文件末尾加上如下代码。关闭后从vscode中再次导出pdf，公式即可正常显示。
# MAGIC ```
# MAGIC <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
# MAGIC <script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>
# MAGIC ```
# MAGIC 
# MAGIC - <font color='red'>Vscode Markdown导PDF</font>：上一条解决方案，在复杂公式中仍然会出错（如：单边大括号，矩阵等）。经多轮实验，有效解决方案如下：
# MAGIC     - 在vscode中，安装拓展包‘markdown preview enhanced’, 不要用默认的preview或其他拓展包的preview功能。
# MAGIC     - 通过‘markdown preview enhanced’中的显示内容，确认语法是否有误，进行调整
# MAGIC     - 语法确认无误后，右键单击preview的页面，选择‘open in browser’，在浏览器中打开内容。浏览器中显示的内容格式与preview中显示的是完全一样的
# MAGIC     - 在浏览器中选择打印功能（ctrl+p），选择‘导出为pdf’，即可将浏览器中看到的内容格式，完全保存到pdf中。（不要选microsoft print to pdf，仍然可能会有格式识别错误）
# MAGIC     - 另注：Markdown PDF第三方插件的PDF导出功能，不能识别复杂公式（包括网上说的改.vscode template的方法，效果都不够）。pandoc第三方插件功能有报错。
# MAGIC 
# MAGIC - 参考：https://blog.csdn.net/oYinHeZhiGuang/article/details/119943358