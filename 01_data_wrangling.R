# =============================================================================
#  01_DATA_WRANGLING.R
#  Modelos de Predicción Analítica — Postgraduate Course
# =============================================================================
#
#  "Data scientists spend 80% of their time cleaning data.
#   The other 20% is complaining about it."  — classic data science joke
#
#  In practice, raw data is almost never ready for modelling.
#  This script teaches you to reshape, filter, and transform data frames
#  using the  tidyverse  ecosystem — the industry standard in R.
#
#  PACKAGES NEEDED:
#    install.packages("tidyverse")   # run once in your console
#
#  DATASET USED:
#    nycflights13::flights  — 336,776 flights departing New York in 2013
#    (install.packages("nycflights13") if needed)
#
#  SECTIONS:
#    1.  The pipe operator  |>
#    2.  filter()    — keep rows
#    3.  select()    — keep / drop columns
#    4.  arrange()   — sort rows
#    5.  mutate()    — create / transform columns
#    6.  summarise() + group_by()  — aggregate statistics
#    7.  Joins       — combining two data frames
#    8.  tidyr       — reshaping: pivot_longer / pivot_wider
#    9.  String manipulation with stringr
#    10. Putting it all together: a real wrangling pipeline
# =============================================================================

library(tidyverse)
# install.packages("nycflights13")
library(nycflights13)


# ── Quick look at the dataset ─────────────────────────────────────────────────

glimpse(flights)   # tidyverse version of str()  — always start here
head(flights)
nrow(flights)      # 336,776 rows
ncol(flights)      # 19 columns


# =============================================================================
# SECTION 1 — THE PIPE OPERATOR  |>
# =============================================================================
#
#  The pipe  |>  (base R, v4.1+) passes the LEFT side as the FIRST argument
#  to the RIGHT side. It lets you read code left-to-right, top-to-bottom
#  instead of inside-out.
#
#  Without pipe (hard to read):
#    round(sqrt(abs(mean(c(-4, 9, 16, -25)))), 2)
#
#  With pipe (reads like a recipe):
#    c(-4, 9, 16, -25) |> mean() |> abs() |> sqrt() |> round(2)

c(-4, 9, 16, -25) |> mean() |> abs() |> sqrt() |> round(2)

#  NOTE: You may also see  %>%  (from the magrittr package / older tidyverse).
#  They behave identically for almost all use cases in this course.
#  We use  |>  as it requires no extra package.


# =============================================================================
# SECTION 2 — filter(): KEEP ROWS THAT MATCH A CONDITION
# =============================================================================
#
#  filter() keeps rows where the condition is TRUE.
#  Think of it as Excel's "Filter" button, but written as code.

# Flights in January:
jan_flights <- flights |> filter(month == 1)
nrow(jan_flights)   # 27,004

# Flights in January AND from JFK:
jan_jfk <- flights |> filter(month == 1, origin == "JFK")
nrow(jan_jfk)

# Flights delayed MORE than 2 hours on departure:
very_late <- flights |> filter(dep_delay > 120)
nrow(very_late)

# Flights to Miami OR Dallas:
mia_dal <- flights |> filter(dest %in% c("MIA", "DAL"))
nrow(mia_dal)

# Flights that were NOT cancelled (dep_time is not NA):
completed <- flights |> filter(!is.na(dep_time))
nrow(completed)

# ── COMMON COMPARISON OPERATORS ───────────────────────────────────────────────
#   ==   equal to              !=   not equal to
#   >    greater than          >=   greater than or equal
#   <    less than             <=   less than or equal
#   %in% value is in a vector
#   is.na()  checks for missing values
#   !    negation (NOT)
#   &    AND (both must be true)
#   |    OR  (at least one must be true)


# =============================================================================
# SECTION 3 — select(): KEEP OR DROP COLUMNS
# =============================================================================

# Keep only carrier, flight number, origin, destination, and delays:
delays_only <- flights |>
  select(carrier, flight, origin, dest, dep_delay, arr_delay)

head(delays_only)

# Drop columns you don't need (prefix with -):
no_times <- flights |>
  select(-year, -hour, -minute, -time_hour)

ncol(no_times)

# Select a range of adjacent columns:
flights |> select(year:day) |> head()

# Select columns whose names start with "dep":
flights |> select(starts_with("dep")) |> head()

# Select columns whose names end with "delay":
flights |> select(ends_with("delay")) |> head()

# Select columns whose names contain "arr":
flights |> select(contains("arr")) |> head()

# Rename a column while selecting it:
flights |>
  select(airline = carrier, departure_delay = dep_delay) |>
  head()

# Rename without dropping other columns — use rename():
flights |>
  rename(departure_delay = dep_delay, arrival_delay = arr_delay) |>
  head()


# =============================================================================
# SECTION 4 — arrange(): SORT ROWS
# =============================================================================

# Flights sorted by departure delay, ascending (smallest first):
flights |>
  select(carrier, flight, dep_delay) |>
  arrange(dep_delay) |>
  head(10)

# Descending order — wrap with desc():
flights |>
  select(carrier, flight, dep_delay) |>
  arrange(desc(dep_delay)) |>
  head(10)    # worst delays ever recorded

# Sort by multiple columns:
flights |>
  select(month, day, dep_delay) |>
  arrange(month, day, desc(dep_delay)) |>
  head(10)

# NOTE: NA values always sort to the END in R, regardless of direction.


# =============================================================================
# SECTION 5 — mutate(): CREATE OR TRANSFORM COLUMNS
# =============================================================================
#
#  mutate() adds new columns (or overwrites existing ones).
#  The new column is computed row-by-row from existing columns.

# Convert air time from minutes to hours:
flights |>
  select(air_time, dest) |>
  mutate(air_hours = air_time / 60) |>
  head()

# Calculate total delay (departure + arrival):
flights |>
  select(dep_delay, arr_delay) |>
  mutate(total_delay = dep_delay + arr_delay) |>
  head()

# Create a categorical column with if_else():
flights_cat <- flights |>
  mutate(
    delay_status = if_else(dep_delay > 0, "Delayed", "On time")
  )

table(flights_cat$delay_status)

# Multiple new columns in one mutate() call:
flights_enriched <- flights |>
  mutate(
    air_hours      = air_time / 60,
    speed_mph      = distance / air_hours,
    total_delay    = dep_delay + arr_delay,
    delay_category = case_when(
      dep_delay <= 0  ~ "Early / On time",
      dep_delay <= 30 ~ "Minor delay",
      dep_delay <= 60 ~ "Moderate delay",
      TRUE            ~ "Major delay"       # TRUE acts as "else"
    )
  )

flights_enriched |>
  select(carrier, flight, dep_delay, delay_category, speed_mph) |>
  head(10)

# transmute() — like mutate() but keeps ONLY the new columns:
flights |>
  transmute(
    air_hours = air_time / 60,
    speed_mph = distance / air_hours
  ) |>
  head()


# =============================================================================
# SECTION 6 — summarise() + group_by(): AGGREGATE STATISTICS
# =============================================================================
#
#  summarise() collapses many rows into ONE summary row.
#  group_by() splits data into groups so summarise() works PER GROUP.
#  This combination is the workhorse of exploratory analysis.

# Overall summary:
flights |>
  summarise(
    total_flights   = n(),                          # n() counts rows
    mean_dep_delay  = mean(dep_delay, na.rm = TRUE),
    median_dep_delay= median(dep_delay, na.rm = TRUE),
    max_dep_delay   = max(dep_delay, na.rm = TRUE),
    pct_delayed     = mean(dep_delay > 0, na.rm = TRUE) * 100
  )

# ALWAYS use  na.rm = TRUE  in aggregation functions to skip missing values.

# Average delay by airline carrier:
by_carrier <- flights |>
  group_by(carrier) |>
  summarise(
    flights         = n(),
    mean_dep_delay  = mean(dep_delay, na.rm = TRUE),
    pct_delayed     = mean(dep_delay > 0, na.rm = TRUE) * 100
  ) |>
  arrange(desc(mean_dep_delay))

by_carrier

# Delay by month (seasonality check):
by_month <- flights |>
  group_by(month) |>
  summarise(
    mean_dep_delay = mean(dep_delay, na.rm = TRUE)
  )

by_month   # July & December tend to be worst

# Group by multiple variables — carrier × month:
carrier_month <- flights |>
  group_by(carrier, month) |>
  summarise(
    mean_delay = mean(dep_delay, na.rm = TRUE),
    n_flights  = n(),
    .groups = "drop"      # always add this to avoid "grouped" surprises
  )

head(carrier_month, 20)

# count() is shorthand for group_by + summarise(n()):
flights |> count(carrier, sort = TRUE)
flights |> count(origin, dest, sort = TRUE) |> head(10)


# =============================================================================
# SECTION 7 — JOINS: COMBINING TWO DATA FRAMES
# =============================================================================
#
#  Joins merge two tables based on a common KEY column.
#
#  ┌─────────────┐   ┌─────────────┐
#  │  Table A    │   │  Table B    │
#  │  key | val  │   │  key | val  │
#  └─────────────┘   └─────────────┘
#
#  left_join  : all rows from A, matching rows from B  (most common)
#  right_join : all rows from B, matching rows from A
#  inner_join : only rows that match in BOTH tables
#  full_join  : all rows from both tables
#  anti_join  : rows in A that have NO match in B  (useful for finding gaps)

# nycflights13 has a companion table  airlines  with full carrier names:
head(airlines)  # carrier | name

# Enrich flights with full airline name:
flights_named <- flights |>
  left_join(airlines, by = "carrier")   # join on the "carrier" column

flights_named |>
  select(carrier, name, flight, dep_delay) |>
  head()

# Join with the airports table to get destination city names:
flights_named2 <- flights_named |>
  left_join(
    airports |> select(faa, dest_name = name, lat, lon),
    by = c("dest" = "faa")   # key has different names in each table
  )

flights_named2 |>
  select(name, flight, dest, dest_name, dep_delay) |>
  head()

# anti_join — find flights whose destination is NOT in the airports table:
missing_airports <- flights |>
  anti_join(airports, by = c("dest" = "faa"))

nrow(missing_airports)
unique(missing_airports$dest)   # these codes have no entry in airports


# =============================================================================
# SECTION 8 — tidyr: RESHAPING DATA
# =============================================================================
#
#  TIDY DATA (Hadley Wickham's principle):
#   - Each VARIABLE gets its own COLUMN
#   - Each OBSERVATION gets its own ROW
#   - Each VALUE gets its own CELL
#
#  Most predictive modelling functions expect TIDY (wide) format.
#
#  pivot_longer() : WIDE → LONG  (many columns → two: name + value)
#  pivot_wider()  : LONG → WIDE  (two columns → many)

# Create a small wide table — average delay by carrier per quarter:
delay_wide <- flights |>
  mutate(quarter = paste0("Q", ceiling(month / 3))) |>
  group_by(carrier, quarter) |>
  summarise(mean_delay = round(mean(dep_delay, na.rm = TRUE), 1),
            .groups = "drop") |>
  pivot_wider(names_from = quarter, values_from = mean_delay)

delay_wide   # one row per carrier, one column per quarter

# pivot_longer() — turn it back to long format for plotting:
delay_long <- delay_wide |>
  pivot_longer(
    cols      = starts_with("Q"),   # which columns to collapse
    names_to  = "quarter",          # new column: the old column names
    values_to = "mean_delay"        # new column: the old values
  )

head(delay_long)

# Another common use: separate() splits one column into two
# Example: "2013-01-15" → year, month, day
library(tidyr)

dates <- tibble(date_str = c("2013-01-15", "2013-07-04", "2013-12-25"))

dates |>
  separate(date_str, into = c("year", "month", "day"), sep = "-")

# unite() does the reverse — combines columns into one:
flights |>
  select(year, month, day) |>
  unite("date", year, month, day, sep = "-") |>
  head()


# =============================================================================
# SECTION 9 — STRING MANIPULATION WITH stringr
# =============================================================================
#
#  stringr provides consistent, readable functions for text data.
#  All functions start with  str_  for easy autocomplete.

# Sample character data:
carriers <- airlines$name
head(carriers)

str_length(carriers)               # number of characters
str_to_upper(carriers)             # ALL CAPS
str_to_lower(carriers)             # all lowercase
str_to_title(carriers)             # Title Case

str_detect(carriers, "Air")        # TRUE where pattern found
carriers[str_detect(carriers, "Air")]   # filter: carriers with "Air" in name

str_replace(carriers, "Airlines", "AL")   # replace first match
str_replace_all(carriers, " ", "_")       # replace ALL matches

str_starts(carriers, "United")     # TRUE if string starts with "United"
str_ends(carriers, "Inc.")         # TRUE if string ends with "Inc."

str_sub(carriers, 1, 6)            # extract characters 1 to 6
str_trim("  extra spaces  ")       # remove leading/trailing whitespace
str_squish("too   many   spaces")  # collapse internal spaces too

str_count(carriers, "a")           # count occurrences of "a" in each string
str_pad("42", width = 6, pad = "0") # pad to width: "000042"

# Useful for cleaning messy imported data:
messy <- c("  United  ", "delta", "SOUTHWEST", "American Airlines ")
clean <- messy |>
  str_trim() |>
  str_to_title()
clean


# =============================================================================
# SECTION 10 — A COMPLETE WRANGLING PIPELINE
# =============================================================================
#
#  Real analysis chains many steps together.
#  Goal: produce a clean, model-ready summary of airline performance.

airline_performance <- flights |>

  # 1. Remove cancelled flights (missing dep_time)
  filter(!is.na(dep_time), !is.na(arr_time)) |>

  # 2. Add useful computed columns
  mutate(
    air_hours      = air_time / 60,
    speed_mph      = distance / air_hours,
    total_delay    = dep_delay + arr_delay,
    is_delayed     = dep_delay > 15,      # FAA definition of "delayed"
    quarter        = paste0("Q", ceiling(month / 3))
  ) |>

  # 3. Join to get full airline names
  left_join(airlines, by = "carrier") |>

  # 4. Group by airline and quarter
  group_by(name, quarter) |>

  # 5. Compute summary statistics
  summarise(
    n_flights       = n(),
    pct_delayed     = round(mean(is_delayed, na.rm = TRUE) * 100, 1),
    mean_dep_delay  = round(mean(dep_delay, na.rm = TRUE), 1),
    mean_arr_delay  = round(mean(arr_delay, na.rm = TRUE), 1),
    mean_speed_mph  = round(mean(speed_mph, na.rm = TRUE), 1),
    .groups = "drop"
  ) |>

  # 6. Sort by worst on-time performance
  arrange(desc(pct_delayed))

# Final clean table — ready for modelling or visualisation:
print(airline_performance, n = 30)

# Save it for use in later scripts:
# write_csv(airline_performance, "data/airline_performance.csv")


# =============================================================================
#  KEY DPLYR VERBS — QUICK REFERENCE
# =============================================================================
#
#   ROWS                 COLUMNS               GROUPS
#   filter()  keep rows  select()  keep cols   group_by()   define groups
#   arrange() sort rows  rename()  rename      summarise()  aggregate
#   slice()   row index  mutate()  add/change  ungroup()    remove groups
#                        transmute() add only
#
#   JOINS                RESHAPE (tidyr)       STRINGS (stringr)
#   left_join()          pivot_longer()        str_detect()
#   inner_join()         pivot_wider()         str_replace()
#   anti_join()          separate()            str_trim()
#   full_join()          unite()               str_to_lower/upper/title()
#
# =============================================================================
#  END OF 01_DATA_WRANGLING.R
#  Next: 02_descriptive_stats.R
# =============================================================================
