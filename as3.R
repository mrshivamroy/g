library(ggplot2)
library(dplyr)
library(readxl)

df <- read_excel("Lab-3.xlsx", sheet = "Sheet1")

# Q1
p1 <- ggplot(df, aes(x = Department, fill = Department)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Number of Students by Department", x = "Department", y = "Number of Students") +
  theme_minimal()
print(p1)

avg_marks <- df %>%
  group_by(Department) %>%
  summarize(Avg_Marks = mean(Marks, na.rm = TRUE))

p2 <- ggplot(avg_marks, aes(x = Department, y = Avg_Marks, fill = Department)) +
  geom_col() +
  geom_text(aes(label = round(Avg_Marks, 1)), vjust = -0.5) +
  labs(title = "Average Marks by Department", x = "Department", y = "Average Marks") +
  theme_minimal()
print(p2)

p3 <- ggplot(df, aes(x = Gender, fill = Gender)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Number of Students by Gender", x = "Gender", y = "Number of Students") +
  theme_minimal()
print(p3)

p4 <- ggplot(df, aes(x = Department, fill = Department)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), hjust = -0.5) +
  coord_flip() +
  labs(title = "Number of Students by Department (Horizontal)", x = "Number of Students", y = "Department") +
  theme_minimal()
print(p4)

p5 <- ggplot(df, aes(x = Department, fill = Gender)) +
  geom_bar(position = "dodge") +
  geom_text(stat = "count", aes(label = after_stat(count)),
            position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "Students by Gender and Department", x = "Department", y = "Number of Students") +
  theme_minimal()
print(p5)

# Q2
dept_dist <- df %>%
  count(Department) %>%
  mutate(pct = n / sum(n) * 100,
         label = paste0(Department, "\n", sprintf("%.1f", pct), "%"))

p6 <- ggplot(dept_dist, aes(x = "", y = pct, fill = Department)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  geom_text(aes(label = label), position = position_stack(vjust = 0.5)) +
  labs(title = "Distribution of Students by Department") +
  theme_void()
print(p6)

gender_dist <- df %>%
  count(Gender) %>%
  mutate(pct = n / sum(n) * 100,
         label = paste0(Gender, "\n", sprintf("%.1f", pct), "%"))

p7 <- ggplot(gender_dist, aes(x = "", y = pct, fill = Gender)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  geom_text(aes(label = label), position = position_stack(vjust = 0.5)) +
  labs(title = "Gender Distribution of Students") +
  theme_void()
print(p7)

# Q3
p8 <- ggplot(df, aes(x = Marks)) +
  geom_histogram(binwidth = 10, fill = "lightblue", color = "black") +
  stat_bin(binwidth = 10, geom = "text", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of Student Marks", x = "Marks", y = "Frequency") +
  theme_minimal()
print(p8)

p9 <- ggplot(df, aes(x = Attendence)) +
  geom_histogram(binwidth = 10, fill = "lightgreen", color = "black") +
  stat_bin(binwidth = 10, geom = "text", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of Student Attendance", x = "Attendance (%)", y = "Frequency") +
  theme_minimal()
print(p9)

p10 <- ggplot(df, aes(x = Study_Hours)) +
  geom_histogram(binwidth = 2, fill = "lightcoral", color = "black") +
  stat_bin(binwidth = 2, geom = "text", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of Study Hours", x = "Study Hours", y = "Frequency") +
  theme_minimal()
print(p10)

p11 <- ggplot(df, aes(x = Marks)) +
  geom_histogram(binwidth = 5, fill = "plum", color = "black") +
  stat_bin(binwidth = 5, geom = "text", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Marks Distribution (Reduced Binwidth/More Breaks)", x = "Marks", y = "Frequency") +
  theme_minimal()
print(p11)

cat("Comparison:\nMarks and Attendance histograms have similar shapes. Since higher attendance usually aligns with higher marks, the frequencies distribute similarly.\n")

# Q4
p12 <- ggplot(df, aes(y = Marks)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Overall Distribution of Student Marks", y = "Marks") +
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
print(p12)

p13 <- ggplot(df, aes(x = Department, y = Marks, fill = Department)) +
  geom_boxplot() +
  labs(title = "Marks Comparison by Department", x = "Department", y = "Marks") +
  theme_minimal()
print(p13)

p14 <- ggplot(df, aes(x = Department, y = Attendence, fill = Department)) +
  geom_boxplot() +
  labs(title = "Attendance Comparison by Department", x = "Department", y = "Attendance (%)") +
  theme_minimal()
print(p14)

p15 <- ggplot(df, aes(x = Gender, y = Marks, fill = Gender)) +
  geom_boxplot() +
  labs(title = "Marks Comparison by Gender", x = "Gender", y = "Marks") +
  theme_minimal()
print(p15)

p16 <- ggplot(df, aes(x = Department, y = Study_Hours, fill = Department)) +
  geom_boxplot() +
  labs(title = "Study Hours across Departments", x = "Department", y = "Study Hours") +
  theme_minimal()
print(p16)

attendance_summary <- df %>%
  group_by(Department) %>%
  summarize(Median_Att = median(Attendence, na.rm = TRUE),
            Var_Att = var(Attendence, na.rm = TRUE))

high_med <- attendance_summary$Department[which.max(attendance_summary$Median_Att)]
low_med <- attendance_summary$Department[which.min(attendance_summary$Median_Att)]
max_var <- attendance_summary$Department[which.max(attendance_summary$Var_Att)]

cat("Highest Median Attendance Department:", high_med, "\n")
cat("Lowest Median Attendance Department:", low_med, "\n")
cat("Largest Variation Department:", max_var, "\n")

outliers <- boxplot.stats(df$Attendence)$out
if (length(outliers) > 0) {
  cat("Outliers in Attendance:", paste(outliers, collapse = ", "), "\n")
} else {
  cat("No outliers detected in Attendance.\n")
}

# Q5
p17 <- ggplot(df, aes(x = Study_Hours, y = Marks, color = Department, shape = Gender)) +
  geom_point(size = 4) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
  labs(title = "Relationship: Study Hours vs Marks", x = "Study Hours", y = "Marks") +
  theme_minimal()
print(p17)

p18 <- ggplot(df, aes(x = Attendence, y = Marks, color = Department, shape = Gender)) +
  geom_point(size = 4) +
  geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
  labs(title = "Relationship: Attendance vs Marks", x = "Attendance (%)", y = "Marks") +
  theme_minimal()
print(p18)

cor_study <- cor(df$Study_Hours, df$Marks, use = "complete.obs")
cor_att <- cor(df$Attendence, df$Marks, use = "complete.obs")

cat("Correlation (Study Hours vs Marks):", round(cor_study, 3), "\n")
cat("Correlation (Attendance vs Marks):", round(cor_att, 3), "\n")
cat("Analysis: Both variables show a strong positive correlation with marks. Higher attendance and more study hours both relate to better student performance.\n")

# Q6
df <- df %>% mutate(Student_ID = factor(Student_ID, levels = unique(Student_ID)))

p19 <- ggplot(df, aes(x = Student_ID, y = Marks, group = 1)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point(color = "blue", size = 3) +
  labs(title = "Marks Obtained by Students", x = "Student ID", y = "Marks") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p19)

p20 <- ggplot(df, aes(x = Student_ID, y = Attendence, group = 1)) +
  geom_line(color = "darkgreen", linetype = "dashed", linewidth = 1) +
  geom_point(color = "darkgreen", shape = 15, size = 3) +
  labs(title = "Attendance by Students", x = "Student ID", y = "Attendance (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p20)

p21 <- ggplot(df, aes(x = Student_ID, y = Study_Hours, group = 1)) +
  geom_line(color = "purple", linetype = "dotted", linewidth = 1) +
  geom_point(color = "purple", shape = 17, size = 3) +
  labs(title = "Study Hours by Students", x = "Student ID", y = "Study Hours") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p21)

# Q7
p22 <- ggplot(df, aes(x = Marks)) +
  geom_freqpoly(binwidth = 10, color = "blue", linewidth = 1) +
  geom_point(stat = "bin", binwidth = 10, color = "red", size = 3) +
  labs(title = "Frequency Polygon of Student Marks", x = "Marks Intervals", y = "Frequency") +
  theme_minimal()
print(p22)

p23 <- ggplot(df, aes(x = Attendence)) +
  geom_freqpoly(binwidth = 10, color = "darkgreen", linewidth = 1) +
  geom_point(stat = "bin", binwidth = 10, color = "red", size = 3) +
  labs(title = "Frequency Polygon of Attendance", x = "Attendance Intervals", y = "Frequency") +
  theme_minimal()
print(p23)

p24 <- ggplot(df, aes(x = Study_Hours)) +
  geom_freqpoly(binwidth = 2, color = "purple", linewidth = 1) +
  geom_point(stat = "bin", binwidth = 2, color = "red", size = 3) +
  labs(title = "Frequency Polygon of Study Hours", x = "Study Hours Intervals", y = "Frequency") +
  theme_minimal()
print(p24)

cat("Comparison:\nThe frequency polygons for Marks and Attendance trace highly similar path shapes, reinforcing the observation that students with higher attendance generally score in higher mark intervals.\n")
