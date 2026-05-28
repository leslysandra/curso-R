# =============================================================================
#  BIENVENIDO A R  |  WELCOME TO R
#  Modelos de Predicción Analítica — DIPLOMADO EN MACHINE LEARNING Y CIENCIA DE DATOS
# =============================================================================
#
#  Author  : Ing. Mgr. Lesly Zerna
#  Version : 1.0
#  Purpose : Introductory guide to R, RStudio, and their role in
#            predictive analytics. No prior coding experience required.
#
#  HOW TO USE THIS FILE
#  ---------------------
#  Read each section carefully. Lines starting with "#" are COMMENTS —
#  they are plain text notes that R ignores when running code.
#  To run a line of code, place your cursor on it and press:
#    Windows/Linux : Ctrl + Enter
#    Mac           : Cmd  + Return
# =============================================================================


# -----------------------------------------------------------------------------
# SECTION 1 — WHAT IS R?
# -----------------------------------------------------------------------------
#
#  R is a FREE, open-source programming language designed specifically for
#  statistical computing and data analysis. It was created in 1993 and is
#  today one of the most widely used tools in data science, research, and
#  business analytics.
#
#  Why R for predictive analytics?
#   - Purpose-built for statistics and modelling
#   - Thousands of packages for machine learning, visualization, and data
#     manipulation (all free)
#   - Reproducible: your entire analysis lives in a script anyone can re-run
#   - Used by top universities, research institutions, and companies worldwide
#   - Large, supportive community and extensive documentation


# -----------------------------------------------------------------------------
# SECTION 2 — RSTUDIO: YOUR WORKSPACE
# -----------------------------------------------------------------------------
#
#  RStudio is the Integrated Development Environment (IDE) we use to work
#  with R comfortably. Think of it as Microsoft Word for R code.
#
#  THE FOUR PANELS YOU WILL SEE:
#
#  ┌─────────────────────────┬─────────────────────────┐
#  │                         │                         │
#  │   SOURCE EDITOR         │   ENVIRONMENT / HISTORY │
#  │   (this file lives      │   (objects you create   │
#  │    here)                │    appear here)         │
#  │                         │                         │
#  ├─────────────────────────┼─────────────────────────┤
#  │                         │                         │
#  │   CONSOLE               │   FILES / PLOTS /       │
#  │   (R runs code here,    │   PACKAGES / HELP       │
#  │    shows results)       │   (viewer & file tree)  │
#  │                         │                         │
#  └─────────────────────────┴─────────────────────────┘
# 
# QUICK TOUR OF RSTUDIO:
# - Source (Top Left): Where you write and save your scripts (like this file).
# - Console (Bottom Left): Where the code actually executes.
# - Environment (Top Right): Your workspace. It shows the data you have loaded.
# - Files/Plots/Help (Bottom Right): Where you see files, graphs, and documentation.
#
#  KEY SHORTCUTS TO LEARN NOW:
#   Ctrl/Cmd + Enter   → Run current line or selection
#   Ctrl/Cmd + S       → Save your script
#   Alt + -            → Insert the assignment operator  <-
#   Ctrl/Cmd + Shift+C → Comment / uncomment selected lines
#   Tab                → Autocomplete (use it constantly!)
#   F1 on a function   → Open its Help page

# ------------------------------------------------------------------------------
# UNDERSTANDING YOUR TOOLS: R vs. RSTUDIO
# ------------------------------------------------------------------------------
# Think of R as the engine of a car, and RStudio as the dashboard and steering wheel.
# - R: The programming language that runs the calculations and statistical models.
# - RStudio: The Integrated Development Environment (IDE) that makes R easy to use.
#

# -----------------------------------------------------------------------------
# SECTION 3 — R AS A CALCULATOR (your first lines of code)
# -----------------------------------------------------------------------------

# R evaluates expressions immediately. Run these lines one by one:

2 + 3          # Addition
10 - 4         # Subtraction
6 * 7          # Multiplication
15 / 4         # Division
2 ^ 8          # Exponentiation  (2 to the power of 8)
sqrt(144)      # Square root
abs(-42)       # Absolute value
log(100)       # Natural logarithm
log10(100)     # Base-10 logarithm
exp(1)         # Euler's number  e^1 ≈ 2.718


# -----------------------------------------------------------------------------
# SECTION 4 — VARIABLES: STORING VALUES
# -----------------------------------------------------------------------------
#
#  A variable is a named container for a value.
#  R uses  <-  as its assignment operator (preferred over =).

edad      <- 28
height   <- 1.75          # meters
name     <- "Ana García"  # text (character) — always use quotes
is_grad  <- TRUE          # logical: TRUE or FALSE

# Print a variable by typing its name and running the line:
age
name

# Or use print() / cat() for more control:
print(height)
cat("Student name:", name, "\n")   # \n = new line

# Do arithmetic with variables:
bmi <- 70 / (height ^ 2)
bmi

# Check what type a variable is:
class(age)
class(name)
class(is_grad)


# -----------------------------------------------------------------------------
# SECTION 5 — VECTORS: R'S FUNDAMENTAL DATA STRUCTURE
# -----------------------------------------------------------------------------
#
#  A vector is an ordered list of values of the SAME type.
#  c() stands for "combine" — the most-used function in R.

# Numeric vector:
grades <- c(85, 90, 78, 92, 88, 76, 95)

# Character vector:
students <- c("Ana", "Bruno", "Carla", "Diego", "Elena", "Felipe", "Gloria")

# Logical vector:
passed <- grades >= 80   # creates TRUE/FALSE for each element

# Inspect your vectors:
grades
length(grades)    # how many elements?
passed

# Accessing elements — R uses 1-based indexing (first element = [1]):
grades[1]         # first grade
grades[3]         # third grade
grades[c(1, 3)]   # first and third grades
grades[2:5]       # elements 2 through 5

# Vectorised operations — R applies functions to ALL elements at once:
grades + 2            # add 2 points to every grade
grades * 0.1          # 10 % of each grade
round(grades / 10, 1) # divide all, round to 1 decimal


# -----------------------------------------------------------------------------
# SECTION 6 — BASIC STATISTICS ON A VECTOR
# -----------------------------------------------------------------------------

mean(grades)      # arithmetic mean
median(grades)    # median
sd(grades)        # standard deviation
var(grades)       # variance
min(grades)       # minimum value
max(grades)       # maximum value
range(grades)     # min and max together
sum(grades)       # total sum
summary(grades)   # five-number summary + mean — use this constantly


# -----------------------------------------------------------------------------
# SECTION 7 — DATA FRAMES: YOUR DATA TABLE
# -----------------------------------------------------------------------------
#
#  A data frame is R's equivalent of a spreadsheet.
#  Each COLUMN is a vector (variable); each ROW is an observation.
#  This is the structure predictive models work with.

student_data <- data.frame(
  name    = students,
  grade   = grades,
  passed  = passed,
  stringsAsFactors = FALSE   # always include this line
)

# Explore the data frame:
student_data          # print the whole table
nrow(student_data)    # number of rows (observations)
ncol(student_data)    # number of columns (variables)
dim(student_data)     # rows × columns
str(student_data)     # structure: types + preview — use this always
head(student_data)    # first 6 rows (useful for large datasets)
tail(student_data, 3) # last 3 rows
summary(student_data) # statistics for every column

# Access columns using the $ operator:
student_data$grade
student_data$name

# Access rows and cells using [row, column]:
student_data[1, ]        # entire first row
student_data[ , "grade"] # entire grade column
student_data[2, "name"]  # cell: row 2, column "name"

# Filter rows by a condition:
student_data[student_data$grade > 85, ]
student_data[student_data$passed == TRUE, ]

# Add a new column:
student_data$grade_letter <- ifelse(student_data$grade >= 90, "A",
                             ifelse(student_data$grade >= 80, "B", "C"))
student_data


# -----------------------------------------------------------------------------
# SECTION 8 — FUNCTIONS: THE BUILDING BLOCKS OF R
# -----------------------------------------------------------------------------
#
#  A function takes inputs (arguments), does something, and returns a result.
#  Syntax:  function_name(argument1, argument2, ...)
#
#  You can also WRITE your own functions — critical for keeping code clean.

# Example: a function that converts Celsius to Fahrenheit
celsius_to_fahrenheit <- function(celsius) {
  fahrenheit <- (celsius * 9/5) + 32
  return(fahrenheit)
}

celsius_to_fahrenheit(0)    # 32
celsius_to_fahrenheit(100)  # 212
celsius_to_fahrenheit(37)   # body temperature ≈ 98.6

# Functions can have multiple arguments and default values:
greet_student <- function(name, language = "EN") {
  if (language == "ES") {
    cat("Bienvenido/a,", name, "— ¡Mucho éxito en el curso!\n")
  } else {
    cat("Welcome,", name, "— Good luck in the course!\n")
  }
}

greet_student("Ana")
greet_student("Bruno", language = "ES")


# -----------------------------------------------------------------------------
# SECTION 9 — PACKAGES: EXPANDING R'S CAPABILITIES
# -----------------------------------------------------------------------------
#
#  R's core is powerful, but its true strength lies in PACKAGES —
#  collections of functions built by the community.
#  CRAN (Comprehensive R Archive Network) hosts 20,000+ packages, all free.
#
#  WORKFLOW:
#   1. Install once  →  install.packages("package_name")    # downloads it
#   2. Load per session →  library(package_name)            # activates it
#
#  ESSENTIAL PACKAGES FOR THIS COURSE:
#
#  Data manipulation:
#   - tidyverse   : umbrella for dplyr, ggplot2, tidyr, readr, purrr, tibble
#   - dplyr       : filter, select, mutate, summarise, arrange data frames
#   - tidyr       : reshape data (pivot_wider, pivot_longer)
#   - readr/readxl: import CSV and Excel files
#
#  Visualisation:
#   - ggplot2     : grammar-of-graphics plotting (publication quality)
#   - plotly      : interactive charts
#   - corrplot    : correlation matrices
#
#  Predictive modelling:
#   - caret       : unified framework for training & evaluating models
#   - tidymodels  : modern tidyverse-native modelling workflow
#   - glmnet      : regularised regression (Ridge, Lasso, Elastic Net)
#   - randomForest: ensemble tree models
#   - xgboost     : gradient boosting (industry standard)
#   - e1071       : Support Vector Machines
#   - rpart       : Decision Trees
#   - nnet        : Neural networks (basic)
#
#  Model evaluation:
#   - yardstick   : metrics (RMSE, MAE, AUC, F1, …)
#   - pROC        : ROC curves and AUC
#
#  Reporting:
#   - rmarkdown   : blend R code + text into HTML / PDF / Word reports
#   - knitr       : dynamic documents


# Install tidyverse (run ONCE — remove the # to uncomment):
# install.packages("tidyverse")

# Load it for this session:
# library(tidyverse)


# -----------------------------------------------------------------------------
# SECTION 10 — A GLIMPSE OF THE PREDICTIVE MODELLING WORKFLOW
# -----------------------------------------------------------------------------
#
#  Every predictive analytics project follows this general pipeline.
#  We will build each step in detail throughout the course.
#
#
#   ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
#   │  1. COLLECT │────▶│  2. CLEAN &  │────▶│  3. EXPLORE   │
#   │     DATA    │     │   PREPARE    │     │   (EDA)       │
#   └─────────────┘     └──────────────┘     └───────┬───────┘
#                                                    │
#   ┌─────────────┐     ┌──────────────┐     ┌───────▼───────┐
#   │ 6. DEPLOY & │◀────│  5. EVALUATE │◀────│  4. BUILD &   │
#   │   MONITOR   │     │   & TUNE     │     │   TRAIN MODEL │
#   └─────────────┘     └──────────────┘     └───────────────┘
#
#
#  STEP 1 — Collect: import CSV, Excel, databases, APIs into R
#  STEP 2 — Prepare: handle missing values, encode categories, scale features
#  STEP 3 — EDA: visualise distributions, correlations, outliers
#  STEP 4 — Model: split train/test, fit model, make predictions
#  STEP 5 — Evaluate: RMSE, MAE (regression) | Accuracy, AUC (classification)
#  STEP 6 — Deploy: export model, build Shiny app, schedule reports

# A minimal end-to-end example using R's built-in dataset  'mtcars':
data(mtcars)         # load dataset: fuel efficiency of 32 cars
head(mtcars)         # preview
summary(mtcars$mpg)  # target variable: miles per gallon

# Visualize the relationship (A simple scatterplot)
plot(cars$speed, cars$dist, 
     main = "Speed vs. Stopping Distance", 
     xlab = "Speed (mph)", 
     ylab = "Stopping Distance (ft)",
     col = "darkblue", pch = 16)

# Split into training (80%) and test (20%) sets:
set.seed(42)                                   # ensures reproducibility
n         <- nrow(mtcars)
train_idx <- sample(1:n, size = 0.8 * n)
train     <- mtcars[train_idx, ]
test      <- mtcars[-train_idx, ]

# Fit a simple linear regression model:
model <- lm(mpg ~ wt + hp + cyl, data = train)  # mpg ~ weight + horsepower + cylinders
summary(model)                                   # coefficients, R², p-values

# Predict on the test set:
predictions <- predict(model, newdata = test)
print(paste("Predicted stopping distance for 20mph is:", round(predictions, 2), "feet"))

# Evaluate — Root Mean Squared Error (RMSE):
rmse <- sqrt(mean((test$mpg - predictions) ^ 2))
cat("Test RMSE:", round(rmse, 3), "mpg\n")

# This is the skeleton of every regression problem we will tackle.
# Classification, trees, ensembles, and regularisation follow the same logic.


# -----------------------------------------------------------------------------
# SECTION 11 — GETTING HELP IN R
# -----------------------------------------------------------------------------

# 1. Built-in help pages  (the most important habit to build):
?mean           # opens the help page for mean()
?lm             # help for linear models
help("data.frame")

# 2. Find functions when you don't know the name:
??regression    # search all help pages for the keyword

# 3. See examples from a function's help page:
example(lm)

# 4. Check arguments of a function:
args(lm)

# 5. Online resources (bookmark these now):
#    https://www.rdocumentation.org          — searchable docs for all packages
#    https://r4ds.had.co.nz                  — R for Data Science (free book)
#    https://www.tidymodels.org              — Tidymodels documentation
#    https://stackoverflow.com/questions/tagged/r  — community Q&A
#    https://posit.co/cheatsheets/           — RStudio cheat sheets (print them!)


# -----------------------------------------------------------------------------
# SECTION 12 — GOOD HABITS FROM DAY ONE
# -----------------------------------------------------------------------------

# 1. ALWAYS set a working directory or use RStudio Projects (.Rproj)
#    so file paths work for everyone on the team:
#    Session → Set Working Directory → To Source File Location
#    Or:  setwd("/path/to/your/project")

# 2. SET A SEED before any random operation so results are reproducible:
set.seed(123)

# 3. COMMENT YOUR CODE — your future self will thank you:
#    Who wrote it, why, when, what assumptions you made.

# 4. NAME THINGS CLEARLY:
#    Good : linear_model_v1, train_data, grade_mean
#    Bad  : x, mymodel, df2, temp99

# 5. NEVER hardcode file paths with backslashes on Windows.
#    Use forward slashes: "C:/Users/Ana/data/students.csv"
#    Or better: use here::here("data", "students.csv")

# 6. ONE SCRIPT = ONE PURPOSE.
#    01_import.R → 02_clean.R → 03_eda.R → 04_model.R → 05_evaluate.R

# 7. RESTART R OFTEN (Session → Restart R) to confirm your script
#    runs from a clean slate without hidden dependencies.


# =============================================================================
#  END OF WELCOME TUTORIAL
#  You are now ready for  01_data_structures.R
#  See you in class!
# =============================================================================
