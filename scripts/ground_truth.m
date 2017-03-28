%First Import Column6 as col vector using matlab import data utility
max_flt = max(VarName6);
min_flt = min(VarName6);
sum_flt = sum(VarName6);
average_flt = mean(VarName6);
std_flt = std(VarName6);
med_flt = median(VarName6);
q1_flt=quantile(VarName6,0.25);
q3_flt=quantile(VarName6,0.75);

int_v= floor(VarName6);
max_int = max(int_v);
min_int = min(int_v);
sum_int = sum(int_v);
average_int = mean(int_v);
std_int = std(int_v);
med_int = median(int_v);
q1_int=quantile(int_v,0.25);
q3_int=quantile(int_v,0.75);