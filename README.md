# Portofolio ADS minor
### Name: Jesse de lange
### Student number: 19043856

In this portofolio, I will show the progress I made during the minor Applied Data Science. During this minor I worked on two different projects:
* FoodBoost
* Cofano Containers

# <a id="contents"></a>Contents<!-- omit in toc -->
- [Individual tasks and reflection](#reflection)
  - [Datacamp certificates](#datacamp-certificates)
  - [Personal contribution to the projects](#personal-contribution-to-the-projects)
  - [Evaluation on group](#evaluation-on-group)
- [The Projects](#the-projects) 
  - [Foodboost project](#foodboost-project) 
  - [Cofano container project](#Cofano-container-project)
- [Predictive analytics](#predictive-analytics) 
  - [Foodboost model](#Foodboost-predictive-model) 
  - [Cofano model](#Cofano-predictive-model)
- [Domain knowledge](#domain-knowlegde)
  - [Literature research](#literature-research)
  - [Terminology, Jargon and Definitions](#terminology-jargon-and-definitions)
- [Data preprocessing](#data-preprocessing)
  - [Foodboost data](#foodboost-data)
  - [Cofano data](#Cofano-data)
- [Communication](#communication)
  - [Presentations](#presentations)
  - [The paper](#paper)

# <a id="reflection"></a>Individual tasks and reflection<!-- omit in toc -->
## <a id="datacamp-certificates"></a>Datacamp certificates <!-- omit in toc -->
* Introduction to Python  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 1.png" />
  </details>
* Intermediate Python  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 2.png" />
  </details>
* Python Data Science Toolbox (Part 1)  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 3.png" />
  </details>
* Python Data Science Toolbox (Part 2)  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 4.png" />
  </details>
* Statistical Thinking in Python (Part 1)   
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 5.png" />
  </details>
* Machine Learning with scikit-learn  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 6.png" />
  </details>
* Linear Classifiers in Python  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 7.png" />
  </details>
* Introduction to Data Visualization with Matplotlib  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 8.png" />
  </details>
* Model Validation in Python  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 9.png" />
  </details>
* Data Manipulation with pandas  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 10.png" />
  </details>
* Exploratory Data Analysis in Python  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 11.png" />
  </details>
* Cleaning Data in Python  
  <details>
  <summary>Certificate</summary>
  <img src="Images/Datacamp 12.png" />
  </details> 
  
## <a id="personal-contribution-to-the-projects"></a>Personal contribution to the projects 
At the beginning of this minor first tried to connect and bond with the group as much as possible to come up with a specific role for me.
Overall it was easy to bond with the group since everybody was over-open about their weaknesses and strengths.

During the Foodboost project, I wanted to contribute to the project with a mathematical part for the project. So I came up with an
Linear Programming model to schedule every predicted meal based on the prediction system the group had made. This model filled in the schedule based on a maximum 
amount of calories and separated predicted lists for meals and dinners. For the maximum amount of calories, I contributed a little bit to the research
so we knew what would be the maximum amount of calories for the Linear Programming model.

For the Cofano project, I mainly was focussing on collecting knowledge about reinforcement learning. I did this using Youtube tutorials and documentation of
different libraries needed. This way I could provide useful knowledge to the team so they could make convincing choices during the making of the final model. I also could share this knowledge with other research groups of the minor. Below I will reflect on the making of my reinforcement learning model.

### **Making a reinforcement learning model by myself** 
#### **Situation** 
To gain knowledge about reinforcement learning and to make every group member more up-to-date on reinforcement learning, we all had to make a reinforcement learning model by our own from scratch with our own unique method.  This so some frustration within the group could be solved, and everybody could contribute an equal amount to the Cofano project.  

#### **Task** 
My task was to make a reinforcement learning model using a [tutorial](https://www.youtube.com/watch?v=Mut_u40Sqz4&t=8980s) I found on youtube, with the necessary [documentation](https://stable-baselines3.readthedocs.io/en/master/) within the four planned weeks.

#### **Action**
I watched the whole 3-hour tutorial on youtube to gain information about the used library “Stable baselines 3”. While doing this, I made notations of all the theoretical matter that was discussed during the video. Meanwhile, I also made the examples given in the video into the GPU Jupyter server. Later on, I made a custom environment given by the video. This explained how to make a regulated temperature model for a shower. When I was done with that, I started applying the model to the Cofano project by changing the state space to a matrix (serving as the spaces a container could be placed), the action space to a multi-discrete (with two integers serving as coordinates for placing a container) and observation spaces also to a matrix of the environment of the model. While doing this I realized I also had to look for the right algorithm to combine it with. I had to choose between two algorithms (PPO and A2C). Eventually, I chose PPO since this was an upgraded version of the A2C algorithm.

#### **Result**
The result of my actions where a very basic working model for a shower temperature regulating model and a container space-filling model. The container model could actually fill in the empty spaces for containers in my environment. I also managed to gain a lot of information about how to use the stable baselines 3 libraries and how to combine the different spaces with the right algorithms of the library. I, later on, used it to share this information with my group for decision-making and share it with other groups.

#### **Reflection**
I am glad that I worked out two models that contained reinforcement learning. This really helped me, the group, and other groups with their further investigation. So I wouldn’t have done anything different regarding gaining information. However, I wasn’t really happy with the model, because I really wanted to expand it with more features like a priority list etc. 

## <a id="personal-learning-objectives"></a>Personel learning objectives
When I started with the minor, I wanted to get an introduction to machine learning and neural networks and how to use them in different situations.
I have always been very drawn to the concept of machine learning and neural networks like making predictions, self-training computers, and recognition. for around half a year I had a book that looked into once in a while, but I had no experience whatsoever.

During this minor, I learned a lot of things about machine learning and neural networks. I learned the basics of fitting and defining a model, data cleaning/preparation, and background knowledge of how such models work. I am also glad that I can do all of this without or with minimal help and use them for my projects at home or later at work. Below I will reflect on hyperparameter tuning the model.  

### **Hyperparametertuning the used model** 
#### **Situation** 
The group made a model and wanted to optimize the results so hyperparameter tuning was needed to make sure all the parameters were correctly defined.

#### **Task** 
Using a study, my task was to find the optimal hyperparameter settings using the optuna hyperparameter tuning method.

#### **Action**
The first thing I did was follow the machine learning lessons of the fourth year of my study. In the lessons, different methods of hyperparameter tuning where explained and I tried them out for myself so I could get a better understanding of the optuna library and how to use it. Then when I had enough information, I applied the method to our model and used it to optimize the results.

#### **Result**
In the end, we got the right hyperparameters and the model eventually worked a little bit better.

#### **Reflection**
It maybe felt like cheating but I was glad that I followed the lectures of my study. This way I gained a lot of knowledge about how the hyperparameters influence the model and how I could optimize them to my demand.

## <a id="evaluation-on-group"></a>Evaluation on group
Overall I'm very happy with my group as a whole. The group existed of people with different domain pieces of knowledge which complimented each other very well. in the beginning, we of course needed to know each other, but eventually, we became a well-oiled machine that was motivated to work on the project.

In the beginning, I really had to find my spot within the group. Of course, I had my role which was to be the communication bridge between the teachers and my group which fittest me well. But within the project, I had to find the right task for myself like anyone else. I also tried to get everything straight for everyone so each one of us knew what the state of the projects was and what needed to be done and came up with ideas to make the workflow more efficient.

Most other teammates were of great use. Martti was a beast in programming and a very motivated student till the end. Joanne really kept the group together during the different meetings we had throughout the minor and Eric really came up with brilliant ideas to make sure the project would be a success. Sefa and Ayrton didn't have much experience with Python so sometimes it was hard for them to keep up. Sefa really did his best with the task he needed to do. He also helped really well when he needed to during the project, despite his non-Python background. Ayrton, however, felt like he was of less use. unfortunately he didn't say that much and we as a group didn't know if he understands it all. We often offered him a helping hand but I don't think he ever was willing to use it. which is a shame since he is a great guy.

For the first 12 weeks, the group gathered as a whole at the THUAS at The Hague on Monday and Wednesday. On Tuesday, and Thursday Joanne, Eric and I gathered at the THUAS at Delft, and Friday we all gathered for the weekly meetings with the teachers at Delft. Monday and Wednesday we had group meetings where we would update the Trello board (see example) and see the results of each individual. For the last 12 weeks, we only gathered at the THUAS at The Hague on Monday when there were lectures, but eventually, we would spend all days at Delft where we would meet as a whole on Friday before we had the meetings with the teachers. Below I will reflect on a problem I solve within the group using an efficient working method.  

<details>
<summary>Trello board example</summary>
<img src="Images/Trello board image.png" />
</details>

### **Working method** 
#### **Situation** 
Within the group, there were a lot of frustrations because people couldn't catch up with each other and couldn't contribute as much to the project as they wanted.  

#### **Task** 
My task was to make sure that everyone would stay motivated for the projects and to make sure that they could keep up with what was happening.

#### **Action** 
I had a couple of conversations with members of the group and what they were the most frustrated about. Then I discovered that they couldn't keep up as much as they wanted. So I thought of a plan to make it work between the group members and came up with the idea to work parallel where anyone would make their own reinforcement learning model.

#### **Result**
The result is that anyone would work parallel on their own reinforcement learning model so that anyone could contribute as much to the project as they wanted. This led to 4 different types of useful models that we could use on the COFANO problem. We also gained a lot of knowledge about the different methods and this was of great value.

#### **Reflection** 
I am glad that the idea of working parallel really worked. I am certainly going to use this in future research when I am working in groups.

# <a id="the-projects"></a>The Projects
## <a id="foodboost-project"></a>Foodboost project
### *Task definition*
In the present day, people are becoming more conscious of what they are eating. This can express itself in eating more environmentally friendly food such as pescatarian, vegetarian and vegan diets. But also in eating more healthily by making fresh and/or biological food. Sometimes it can be easy to put together a diet that fits a person the most, but often it can be a really hard task. The goal of the foodboost project is to make a recommendation system that predicts if someone likes a certain type of food and implement this into an easy-to-make and maintain diet.

During this project, me and my group members mainly focussed on the basis for the predicting part of the project and fitting the predicted dishes into someone's diet in consideration of a nut allergy. To make progress we defined the following research questions:

    With what method can we make weekly lunch and dinner recommendations considering a nut allergy or intolerance, with consideration of a combined average amount of calories for lunch and dinner combined?

    1. What food can be considered a nut allergen?
    
    2. What is the average amount of calories for lunch and dinner combined?

    3. What method is used/can be used for predicting if someone likes a lunch or dinner?

    4. What method can be used to make a diet schedule with as much variety as possible?

With question 1 we wanted to make sure that we could filter out every lunch/dinner that consisted of ingredients that contained or are a nut allergen. This was a big part of processing the data so no food with a nut allergen could be selected during the prediction and during the making of a diet schedule.

With question 2 we wanted to find out how many calories an average person ate during lunch and dinner combined. This would come in handy when we would schedule the predicted lunch and dinners a person would like. We of course wanted this to be a healthy diet, so a maximum amount of calories would be a good restriction.

With question 3 we wanted to research the different options for a prediction model for our recommendation system.

With question 4 we did research on the different methods used to fit in a schedule with the predicted lunch and dinners. We wanted to make sure to recommend a schedule that has a big variety of lunches and dinners.  

### *Evaluation*
With our research, I think we have laid a good basis for a better and more efficient model. I think if someone would pick up this project, there could be a few interesting pick-up points to start:

The first point where there could be made a lot of improvement is the prediction system. Right now our system consisted of a simple machine-learning decision tree, which does the job but is not really efficient enough. I would recommend researching somewhere in the range of neural networks. I think you could come to a far end with using embeddings, which also are used to make recommendation systems for movies.

The second biggest point of improvement would be to make more restrictions to the now-used Linear Programming model. Right now the model gives a diet schedule with more variety in lunches and dishes. It would be a great idea if there would be some research on the different kinds of kitchens around the world and also make it variate in that way. Now we have a chance of having 7 kinds of pasta every week which makes sure that you will not eat the same dish every night, but not that you variate between different kitchens.

### *Conclusions*
After 6 weeks we managed to answer our research question. As explained the research question was divided into 4 subquestions leading to the main answer.

#### What food can be considered a nut allergen?
One of the first things we did was search on the internet for a list or table of different ingredients that were considered a nut allergens. As we had found different ingredients that contained nuts, we eventually made a list of different kinds of nuts and filter them out. Also, we made a `false-positives` list where we would put in the different kinds of ingredients that did not contain any form of nuts, but in the word itself contained `noot` (i.e. Kokosnoot, Nootmuskaat). With those lists, we iterated over each recipe in the Allerhande dataset and looked if the ingredients contained anything that could be considered a nut allergen.

#### What is the average amount of calories for lunch and dinner combined?
The average amount of lunch and dinner combined for a day came out to be 1040 Kcal a day. This number was determined after looking at the following graphs and approaching them with basic statistics:

<details>
<summary>Statistical graphs</summary>
<img src="Images/Foodboost Kcal Graphs.png" />
</details>

#### What method is used/can be used for predicting if someone likes a lunch or dinner?
For the predictions, we had to deal with a classification problem. So we chose to use the decision tree classifier for a predictive machine learning model. To use this we had to make binary dummy variables for the ingredients used in a meal. We chose the decision tree classifier because compared to other models, this gave us a higher overall score.

<details>
<summary>Model comparison</summary>
<img src="Images/Foodboost Model comparison.png" />
</details>

The next thing we did was using and optuna study for hyperparameter tuning to tune the model to achieve a higher precision score:

<details>
<summary>Optuna study results</summary>
<img src="Images/Foodboost hyperparameter tuning results.png" />
</details>


#### What method can be used to make a diet schedule with as much variety as possible?
To schedule the lunches and dinners in a varied way, we used a linear programming model using the `ORTools.linear_solver` library. This model had a total of 4 restrictions.

- It could fill in a maximum of 1040 Kcal a day for lunch and dinner combined 
- It can use only 1 lunch a day
- It can only use 1 dinner a day
- It can only select lunch or dinner once

We did not use any restrictions for the maximum amount of calories for lunch and dinners separately, because we wanted to be flexible in our schedule so more lunches and dinners could be selected for the diet schedule. We made a plot showing the scheduling results of de linear programming model and the distribution of the lunch and dinner Kcal each day and the corresponding schedule:

<details>
<summary>Diet schedule</summary>
<img src="Images/Foodboost diet schedule .png" />
</details>

<details>
<summary>Lunch and dinner kcal distribution</summary>
<img src="Images/Foodboost lunch and dinner distribution.png" />
</details>

### *Planning*
Since this was our first time working with Trello, almost all of the group members had to get used to Trello. We tried to have a weekly meeting where we would discuss our tasks for the week where we would set deadlines for the tasks and document the tasks on the Trello board. Eventually, this wasn't maintained during the Foodboost project, since some changes in our project had to be made fast and we lost focus on the Trello workstyle in the last couple of weeks.

Nevertheless, we all had our own tasks and we set the deadline for our tasks on the day of the external presentation. This seemed to be working since we had a model that worked surprisingly well. 

<details>
<summary>Foodboost Trello board</summary>
<img src="Images/Foodboost planning trello.png" />
</details>

[Back to contents](#contents)

## <a id="Cofano-container-project"></a>Cofano container project
### *Task definition*

At the port of Rotterdam, many ships come and go with their cargo. Shipping is an important part of logistics with many problems left to solve. Cofano is one of the companies that is working on a method to automate container stacking problem and solve the container stacking problem. 

Cofano is still researching methods to tackle the container stacking problem and asked us to help them. The main goal of tackling the container stacking problem is to minimize the steps from moving containers from ship to ship. Every step a container (or better said, reach stacker) takes to move the containers is called a `move` which we want to minimize. Minimizing these `moves` can help reduce unnecessary costs and delays. 

Our goal is to find a method that Cofano can use to automate the container stacking problem. To do this, we set up a few questions we can answer to help us find a suitable method for this problem:

    With what method can we solve the container stacking problem using only one reach stacker and focusing only on unloading one ship with a random load? 

    1. What methods are available to solve the container stacking problem?
    
    2. What is a move, and what are the restrictions?

    3. What type of containers do we have to use?

    4. What is the layout of the port?
    
    5. How can we simulate container data?

With question 1 we wanted to orientate on the different possible methods that can help solve the container stacking problem.

With question 2 we wanted to define the word `move` so we knew what the main goal of Cofano was. Also, we wanted to define some restrictions for the method were going to use, so the model gives feasible solutions.

With question 3 we wanted to know the different types of containers since the cargo can contain different types of containers and sizes.

With question 4 we wanted to know the port layout, so we can fit our model specifically to the situation that Cofano is dealing with.

With question 5 we wanted to know how we can create our data, this was a very important part of our research since the Cofano dataset was very confusing.

### *Evaluation*
For the container stacking problem, we came up with a way to use reinforcement learning to make a layout for unloading the containers onto the dock. However, we think there are a couple of good starting points for future research to improve this model since we only got the tip of the iceberg here:

The first thing for improvement is to expand the layout of the dock in the model. Right now the space in the model assumes there is a one-level container space, but in reality, it of course would be interesting to be able to fill in more layers.

The second point of improvement is working on the priorities of the containers. Right now we have attached 4 different kinds of priorities to containers, but in the real world, it would be nicer to implement more priorities and even container IDs to the problem to fill in the containers in an efficient way. 

The third thing I would recommend is using an operations research method to take into account the driving time of the reach stacker. If you can find a way to also minimize the traveling time of the reach stacker to a place on the dock, you can unload the ship faster which is more financially interesting. 

The fourth thing I can suggest is applying different containers to the model in then using a different kind of measure to determine the placing zone restrictions for the model.

The last thing I would recommend is implementing more reach stackers and or cranes, based on the availability of the port Cofano is working on.

### *Conclusions*
For the last 14 weeks, we worked on the container stacking problem. Where we made a model that trains to make a layout on a dock using reinforcement learning. While doing this we used different research questions to achieve our goals. 

#### What methods are available to solve the container stacking problem?
For solving the container stacking problem we used a reinforcement learning model, which can train using its own generated data he got from previous experiences. The model uses an agent and environment. The model also makes use of a Deep Q Network (DQN) algorithm to solve each episode.

#### What is a move, and what are the restrictions?
A move is placing a container on the dock. The restrictions are as follows:

- We can only move one container at a time.
- We can only pick up containers at the long sides. This means that we can only pick up containers on two sides of the dock.
- A container can not be placed in an empty space
- A container stack can not be higher than 5 containers


#### What type of containers do we have to use?
We only use one size for a container. which means we don't have to worry about the exact sizes or types. 

#### What is the layout of the port?
The port layout is as follows:

<details>
<summary>Satellite pictures</summary>
<img src="Images/Containers satalite pictures.png" />
</details>

<details>
<summary>Container zone mapping</summary>
<img src="Images/Containers container mapping.png" />
</details>

However, we didn't use the layout since we decided to begin on a small scale and didn't have any time to expand the container zones in our model because of finetuning.

#### How can we simulate container data?
Using reinforcement learning we didn't have to make new data like the dataset we have been given. The only thing we have done is make a random list of high and low priorities (using numbers 1 to 4 ) to simulate a ship. 

### *Planning*
During this project, we found a more efficient way of working with Trello. The first thing we did was set up a more efficient and clear Trello board so we could work this out much more. 

We created different kinds of categories within the Trello board so we had a clear understanding of what the project would look like. The second thing we did was assign someone responsible for a certain category. 

Every time we had a group meeting we asked anyone what the status of their task was and immediately updated the Trello board. Also here, some tasks had deadlines and others were optional. I am more satisfied with the way we worked with Trello. I think, that the experience of the foodboost project was a big reason why it worked better this time.

We also made an Excel project planning to give an overall view of the necessary steps we had to take during the project.

<details>
<summary>Container Excel planning</summary>
<img src="Images/Excel planning.png" />
</details>

<details>
<summary>Container Trello board</summary>
<img src="Images/Trello board image.png" />
</details>

[Back to contents](#contents)

# <a id="predictive-analytics"></a>Predictive analytics <!-- omit in toc -->
## <a id="Foodboost-predictive-model"></a>Foodboost predictive model
For the foodboost project, we had to make a recommendation system that could recommend a certain variate diet. This problem consisted of a classification problem (in the form of 1, which means yes, and 0 which means no). The dataset we eventually used contained if someone liked a certain meal, what the name of the meal is and what ingredients the meal contained. The names in this situation were a string and were not used in the prediction model itself. The other variables were binary variables (like I said earlier). 

For this reason, we needed to use a predictive classification model. Due to our inexperience with machine learning at the time, we used the first predictive classification models we could think of:

- Logistic regression classifier
- Decision tree classifier
- K Nearest Neighbors classifier

In the following notebook, I contributed to selecting the model by making a small table containing the 3 different models and their `f1_score`, `precision_score`, `recall_score`, and `accuracy_score`. 

<details>
<summary>Model comparison</summary>

| Model                  | Recall    | Precision| Accuracy | F1 score |
|------------------------|-----------|----------|----------|----------|
| Logistic Regression    | 0.769231  | 0.714286 | 0.833333 | 0.660377 |
| DecisionTreeClassifier | 0.853933  | 0.904762 | 0.808511 | 0.754717 |
| KNeighborsClassifier   | 0.827586  | 0.857143 | 0.800000 | 0.716981 |
</details>

With this model comparison. We decided to investigate further with the `precision_score`. This score mainly focuses on telling if something is a true positive. This means, in the context of our project, that it focuses on predicting that if someone likes it according to the prediction, he actually likes it in real life. And also I laid my eyes on the decision tree classifier, because of the higher `precision_score`. 

Furthermore, I was focussing more on hyperparameter tuning using an optuna study. With the hyperparameter tuning, I tuned the 2 most used parameters `max_depth` and `min_samples_leaf`.

<details>
<summary>Optuna study results</summary>
<img src="Images/Foodboost hyperparameter tuning results FV.png" />
</details>

Also, I helped with predicting the data. I helped set up and fit a decision tree classifier with the chosen hyperparameter settings. which with the rest of the explained code can be checked in the corresponding [notebook](Notebooks/Selecting%20and%20Hyperparameter%20tuning%20model.ipynb).

In the final version, I linked the decision tree model to the linear programming model. I took the initiative to make the linear programming model to further process the predictions of the decision tree model. I manage to go generate a feasible outcome with the four restrictions I highlighted earlier. In the final [notebook](Notebooks/Final%20model%20Foodboost.ipynb). 

The results of the total model:

<details>
<summary>Diet schedule</summary>
<img src="Images/Foodboost diet schedule .png" />
</details>

<details>
<summary>Lunch and dinner kcal distribution</summary>
<img src="Images/Foodboost lunch and dinner distribution.png" />
</details>

I finally wanted to make a simple model on my own using the knowledge I gained from the Machine Learning lectures of my study. So I went to make a model that could predict how many calories were in a certain meal, using the nutrition values of the given Allerhande dataset. 

After the data preprocessing, I First decided to see how each model would do with a certain test and training set I created. For this, I also created a small table where I could see how great different models would predict with the given data. I choose between 2 different models (Ridge and Linear Regression). When I made my function, this table I got:

<details>
<summary>table of model performance</summary>

| Model                  | MSE       | RMSE     | R2_score | 
|------------------------|-----------|----------|----------|
| Linear Regression      | 1.932859  | 1.390273 | 0.945735 |
| Ridge                  | 1.932852  | 1.390270 | 0.945735 |
</details>

Overall the Linear Regression model was slightly better on the MSE and RMSE, but I decided to use the R2_score as a metric for evaluating the model. On this metric, Both models performed the same, so I decided to use the Ridge model since I already had a little experience with the ridge model.

The next thing I did was hyperparameter tuning the model on the hyperparameter alpha. For the hyperparameter tuning, I also used the optuna study. When I ran the optuna study and it was done. I made a plot to determine the best alpha for my model. This time it was a little harder to tell the right hyperparameter value, so I went with the value that seemed to be a little higher (0.05):

<details>
<summary>Lunch and dinner kcal distribution</summary>
<img src="Images/Foodboost hyperparameter tuning own experiment.png" />
</details>

In the end, when my decision-making was done, I fitted the predictive model to my dataset and predicted my test set. When I had the predicted values I quickly evaluated them by comparing them with the real values of the dataset. I was very glad about the results since the R2_score of my model come down to around 0.906. I thought this was a great achievement! See the following notebook for the results:

[Notebook](Notebooks/Foodboost%20datacleaning%20and%20model%20jesse.ipynb)

[Back to contents](#contents)

## <a id="Cofano-predictive-model"></a>Cofano predictive model

For the Cofano project, we decided to fully focus ourself on the reinforcement learning method. For this method, I made a model myself using the [Stable Baseline 3 library](https://stable-baselines3.readthedocs.io/en/master/). I first wanted to have an introduction and used a [Youtube instruction video](https://www.youtube.com/watch?v=Mut_u40Sqz4&t=8980s) to do so. 

With Stable baselines 3 and the video I managed to make a very basic balancing reinforcement model, which uses a shower temperature regulating model as seen in the shower model notebook:

[Notebook](Notebooks/Reinforcement%20model%20Jesse%20shower.ipynb)

Later on, I decided to transform the made model from the introduction video by changing the observation space (to a Box/Matrix) with each variable in the matrix serving as a location where the containers could be placed, action space (to a MultiDiscrete) which served as coordinates to place a container into the matrix and state of the model(to a Box/Matrix). With this, I had to take into consideration that I had to look up what algorithm I had to use within my model. The shapes of the previous mentions variables were of great influence on this choice:

<details>
<summary>Algoritmns with space requirements</summary>
<img src="Images/Containers table of reinforcement learning algoritms.png" />
</details>

Since all algorithms were capable of handling all observation space shapes. I had to make my choice based on the action space shape. Since only A2C and PPO could work with a MultiDiscrete action space, I did a little experiment with both of the algorithms on the container problem. After running both models I saw that the A2C algorithm learned much better with filling in containers to non-occupied places in the matrix over 57 seconds. With this said I wanted to go with A2C as my algorithm for learning. You can see the learning curve for the 2 algorithms here:

<details>
<summary>A2C en PPO Learning curve</summary>
<img src="Images/Containers A2c en PPO Learning curve V2.png" />
</details>

And see my expirimentation in this notebook:
[Notebook](Notebooks/_Reinforcement%20model%20Jesse%20containers.ipynb)

The next thing I did for the Cofano project was investigating if it was possible to use a CNN for our reinforcement learning model with Martti. With this model which is normally used to identify pictures, we wanted to use the tensors (which are normally pixels in an image) as places for our containers. This could come in handy if you want to train a model where it can identify the different priority placement regions for the containers. And for the (normally used) RBG layers of a picture, we could fill in the different placement layers for the containers. In the notebook below, you can see that I made the CNN class. Here Martti and I experimented with the Conv2D (used for Images) and Conv3D (used for a sequence of images, or video) setups for the model. Furthermore, I gave Martti some theoretical information about how to use a CNN and what all the different variables mean.

[Notebook](Notebooks/CNN%20Reinforcement%20V2%20Row%20Environment.ipynb)

[Back to contents](#contents)

# <a id="domain-knowlegde"></a>Domain knowledge <!-- omit in toc -->
The foodboost project is based on the nutrition field of research mixed with data science. Luckily, this subject field is on average not very strict if you are looking at the project we did. However, we did incorporate nut allergies into our research which means there is no room for error since a wrong filtering system could have serious consequences for the person using the recommendation system. For our project, this also means that we need to know the steps where the nut allergen filtering happens. The prediction part of our project doesn't have these strict rules since the worst thing that could happen is that someone doesn't like the food the system recommended.

The Cofano project's subject field is based on operations research mixed with data science. With this project, we had to make an efficient layout for the incoming containers from ships. This means that we have to set restrictions for our model based on a real-life process. The main goal of this subject field is to make processes as efficient as possible, to make sure the probability of delay is reduced, and to make sure the costs of the process are minimalized. Within this subject field, gaining more efficiency from a model is key. So every feasible improvement is a win. However, making the model feasible does contain some strict checks to make sure it is possible.

## <a id="literature-research"></a>Literature research

For the foodboost project, I used this documentation to gain information for the project:
- [Introduction to Machine Learning with Python by Sarah Guido](https://www.bol.com/nl/nl/p/introduction-to-machine-learning-with-python/9200000036116281/?bltgh=vxHmywO-ITJATpEgw-BLQQ.2_16.17.ProductTitle)
- [OR tools library documentation](https://mlabonne.github.io/blog/linearoptimization/)

For the cofano project I used the following documentation and information sources:
- [OR tools library documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [Tutorial about reinforcement learning](https://www.youtube.com/watch?v=Mut_u40Sqz4&t=8980s)

## <a id="terminology-jargon-and-definitions"></a>Terminology, Jargon and Definitions
<details>
<summary>List of definitions</summary>

- Allergen: a specific type of nutrition that can cause allergic symptoms
- Linear programming: A linear mathematical approach within the operations research that can be used to optimize certain processes 
- Reinforcement learning: A neural network that can learn from self-generating data, it generates the data by doing actions in an environment with an agent
- Environment: A digital space that gives feedback to an agent
- Agent: An agent decides what action to take using the outcome of the previous steps the agent took and the current status of the environment
- Move: A displacement of a container from one place to another
- Reacher stacker: A reacher stacker is a vehicle used to move the containers of a ship
- Restriction: A rule that the model has to apprehend.
- Discrete space: an integer
- Box Space: A matrix with integers or floats
- Multidiscrete: A list of discrete numbers

</details>

[Back to contents](#contents)

# <a id="data-preprocessing"></a>Data preprocessing<!-- omit in toc -->
## <a id="foodboost-data"></a>Foodboost data
For the foodboost project, one of my tasks was to orientate on the recommended amount of calories for lunches and dinners separately. To define the maximum amount for both meals, Joanne and I searched around on the internet for a recommended amount of calories each day. Here I created a list with Joanne that contains these numbers in separate lists for lunch and dinner. When these lists were made I made a boxplot for both recommended lunch and diner calories, so we can see clearly how the calories were distributed:

<details>
<summary>Lunch and dinner kcal boxplot</summary>
<img src="Images/Foodboost Kcal Graphs boxplot.png" />
</details>

As you can see, We only really had 2 outliers in the lunch calorie boxplot. We also see that the recommended amount of calories for a lunch lies between 450 and 500 Kcal. We see no clear outliers in the boxplot of the diner calories amount. Next up was to visualize the data for lunch and dinner separately. This time we used a barplot to have further insides into the distributions of the calories:

<details>
<summary>Lunch and dinner kcal distributions</summary>
<img src="Images/Foodboost Kcal Graphs 2 graphs.png" />
</details>

Using this image we decided to take around 640 for dinner and around 400 for lunch. This of course was a guideline since we decided to be flexible in the number of calories for lunch and dinner separately so that the model could get more options to fill in the schedules. Our code could be seen below:

[Notebook](Notebooks/Statistiek%20op%20calorie%C3%ABn.ipynb)

Eventually, I wanted to walk the process of cleaning data and predicting all on my own. So I started with a simple predicting model to predict the number of calories in a meal. I first took a look at the dataset to see in what form the dataset was. Then when I was finished, I decided to make a workable data frame out of it. This first seemed a little hard, but with a few tips from Joanne on how to flip the rows and columns I quickly figured it out. 

The next thing I did was fill out all the NaN values in the list to 0. I thought this was a good idea because it would give a workable data frame so I could get an inside of all the columns without wasting too much data (which I could do by deleting the rows that contained a NaN value). Then I created a train and test sample, by splitting the data frame into 2 Dataframes. The first 8000 rows would be part of my data frame and the last 706 rows would be my test set.

The next thing I did was figure out what columns had the biggest correlation with my Y value (energy in kcal). I did this by using a correlation function in python. So when I found my 3 most correlating columns, I started to look if any of those columns correlated with each other. I decided to set the bar to a 0.6 correlation because, at the machine learning lectures of the 4th year of my study, I learned that that was a good measure. Luckily, all the 3 columns did not have a high enough correlation to not take them for predicting

Next up I tried to see if all data needed to be transformed so it was workable for the model. unfortunately, all the data were not normally distributed, so I decided to transform them all with the transformation they needed. For this I used a picture to determine what the original distribution was, to determine the transformation I had to use:

<details>
<summary>Distribution image</summary>
<img src="Images/Foodboost Distributions.png" />
</details>

So when I did this to the dataset I made my predictions. When I was done with my predictions for the number of calories, I of course had to transform back my data to their original shape before the evaluation. You can see the entire data-cleaning/processing here:

[Notebook](Notebooks/Foodboost%20datacleaning%20and%20model%20jesse.ipynb)

## <a id="Cofano-data"></a>Cofano data
For the Cofano project, I didn't do too much data preprocessing. I mainly focussed on getting information on reinforcement learning and how to use it. However, I had to make sure that the environment was using the correct data so I could choose the right algorithm to go with it. So first I decided to take a look at all the different shapes for the action, observation, and state spaces. Then I decided for the Cofano project that I wanted to use a matrix to serve as the locations for the containers, and use the action space as a MultiDiscrete space. With this setup, I could make the model choose some coordinates to place the containers. See the following notebook:

[Notebook](Notebooks/_Reinforcement%20model%20Jesse%20containers.ipynb)

[Back to contents](#contents)

# <a id="communication"></a>Communication<!-- omit in toc -->
## <a id="presentations"></a>Presentations

* [Foodboost week 2](Presentations/foodboost%20-%20Week%202.pdf)
* [Foodboost week 4](Presentations/foodboost%20-%20week%204.pdf)
* [Foodboost external presentation week 6](Presentations/foodboost%20-%20eindpresentatie%20Week%206.pdf)

* [Container week 8](Presentations/container%20-%20Week%208.pdf)
* [Container week 12](Presentations/container%20-%20Week%2012%20.pdf)
* [Container week 14](Presentations/container%20-%20Week%2014.pdf)
* [Container external presentation week 14](Presentations/container%20-%20eindpresentatie%20Week%2014.pdf)

## <a id="paper"></a>The paper
In the paper, I first helped Martti with writing a section about the method we used. Here I wrote about what reinforcement learning is and how it worked. I explained the basic principles of reinforcement learning and gave some examples of how it is used. Then I explained with Martti's help how the agent worked and how it corresponds with the environment. We were then told by Tony that we didn't need to explain the principle of reinforcement learning since it was not our invention. So we decided to briefly mention the reinforcement principle.

Later I made the discussion part of the paper. I thought this was quite challenging since I never wrote a paper. So I first read the results section of the paper and then asked a student who was familiar with writing papers for some tips. He then explained the way a discussion in a paper worked and I wrote the part

The next thing I did was give Ayrton guidelines for writing the conclusion. Here I gave him some bullet points and structure for the conclusion part of the paper. After he wrote it down, I quickly checked the overall message.

At last, I ran through the whole paper with Martti. Here I changed some lines and words to give the paper an overall more fluent story. I also checked if the papers' information was all correct and changed it where ever it was needed.

[Back to contents](#contents)
