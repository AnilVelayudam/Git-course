import numpy as np

# Define the function to convert strings to uppercase
def to_upper(df, col, column_name):
    col_name = column_name[0]
    df[col] = df[col_name].str.upper()
    return df

# Define the function to convert strings to uppercase
def to_lower(df, col,column_name):
    col_name = column_name[0]
    df[col] = df[col_name].str.lower()
    return df

def to_reverse(df, col,column_name):
   col_name = column_name[0]
   df[col] = df[col_name].apply(lambda x: str(x)[::-1])
   return df

# Define the function to convert strings to ASCII values
def to_ascii(df, col,column_name):
    col_name=column_name[0]
    df[col] = df[col_name].apply(lambda x: ord(x))
    # df[col] = df[col_name].apply(lambda x: [ord(char) for char in x])
    return df




def to_substr(df, col,column_name): 
    col_name = column_name[0]
    start = int(column_name[1])
    length = int(column_name[2])
    df[col] = df[col_name].str[start:start+length]
    return df


# Define the RTrim() function
def to_rtrim(df, col,column_name):
    col_name=column_name[0]
    trim_set = column_name[1]
    df[col] = df[col_name].str.rstrip(trim_set)
    return df


# Define the LTrim() function
def to_ltrim(df, col,column_name):
    col_name=column_name[0]
    trim_set = column_name[1]
    df[col] = df[col_name].str.lstrip(trim_set)
    return df


def to_rpad(df, col,column_name):
    col_name = column_name[0]
    length = int(column_name[1])
    char = column_name[2][1] if len(column_name) > 2 else ''
    df[col] = df[col_name].str.pad(width=length, side='right', fillchar=char)
    return df


# Define the Replacestr() function
def to_replacestr(df, col,column_name):
    col_name = column_name[0]
    old_str = column_name[1].strip("'")  # Remove single quotes from old_str
    new_str = column_name[2].strip("'")  # Remove single quotes from new_str
    df[col] = df[col_name].str.replace(old_str, new_str)
    return df



def to_replacechr(df, col,column_name):
    col_name = column_name[0]
    old_char = column_name[1].strip("'")
    new_char = column_name[2].strip("'")
    df[col] = df[col_name].str.replace(old_char, new_char)
    return df


# Define the Lpad() function
def to_lpad(df, col,column_name):
    col_name = column_name[0]
    length = int(column_name[1])
    char = column_name[2][1] if len(column_name) > 2 else ''
    df[col] = df[col_name].str.pad(width=length, side='left', fillchar=char)
    return df


def to_length(df, col,column_name):
    col_name = column_name[0]
    df[col] = df[col_name].str.len()
    return df

def to_instr(df, col,column_name):
    col_name = column_name[0]
    substring = column_name[1].strip("'")  # Remove single quotes from the substring
    df[col] = df[col_name].str.find(substring)
    return df


def to_chr(df, col,column_name):
    col_name = column_name[0]
    df[col] = df[col_name].apply(lambda x: ''.join(chr(int(c)) for c in x.replace(',', ' ').replace('-', ' ').split() if c.isdigit()))
    return df

# Define the to_chrcode function
def to_chrcode(df,col ,column_name):
    col_name = column_name[0]
    df[col] = df[col_name].astype(str).apply(lambda x: ''.join(chr(ord(c) + 1) for c in x))
    return df


# IndexOf: Get the index of a substring in a string
def to_indexof(df,col ,column_name):
    col_name = column_name[0]
    substring = column_name[1].strip("'")
    df[col] = df[col_name].str.find(substring)
    return df
 
def to_initcap(df,col ,column_name):
    col_name=column_name[0]
    df[col] = df[col_name].str.title()
    return df

# Define the function to concatenate columns
def to_concat(df, col, column_names, separator=''):
    # Extracting column names from column_names list
    col1_name, col2_name = column_names
    
    # Concatenating the two columns with the specified separator
    df[col] = df[col1_name].astype(str) + separator + df[col2_name].astype(str)
    
    return df

############Numerical####################################################################################################################

# Define the function to apply absolute value operation((-) values to +postive)
def to_abs(df,col ,column_name):
    col_name=column_name[0]
    df[col] = df[col_name].abs()
    return df

# Define the function to apply ceiling operation

def to_ceil(df,col ,column_name):
    col_name=column_name[0]
    df[col] = np.ceil(df[col_name])
    return df


# Define the function to apply cumulative sum operation
def to_cume(df,col ,column_name):
    col_name=column_name[0]
    df[col] = df[col_name].cumsum()
    return df

def to_exp(df,col ,column_name):
    col_name=column_name[0]
    df[col] = np.exp(df[col_name])
    return df

# Define the function to apply floor operation
def to_floor(df,col ,column_name):
    col_name=column_name[0]
    df[col] = np.floor(df[col_name])
    return df

# Define the function to apply natural logarithm operation with error handling
def to_ln(df, col, column_name):
    col_name=column_name[0]
    df[col] = np.log(df[col_name])
    return df

# Define the function to apply rounding operation
def to_round(df, col, column_name):
    col_name=column_name[0]
    df[col] = np.round(df[col_name])
    return df

# Define the function to apply sign operation Returns whether a numeric value is positive, negative, or 0.
def to_sign(df, col, column_name):
    col_name=column_name[0]
    df[col] = np.sign(df[col_name])
    return df

# Define the function to apply square root operation
def to_sqrt(df, col, column_name):
    col_name=column_name[0]
    df[col] = np.sqrt(df[col_name])
    return df

# Define the function to apply truncation operation
def to_trunc(df, col, column_name):
    col_name=column_name[0]
    df[col] = np.trunc(df[col_name])
    return df

# Define the function to apply logarithm operation with base
def to_log(df, col, column_name):
    col_name=column_name[0]
    df[col] = np.log(df[col_name])
    return df

# Define the function to apply modulus operation
def to_mod(df, col, column_name):
    col_name = column_name[0]
    divisor = int(column_name[1])  # Convert divisor to an integer
    df[col] = df[col_name] % divisor
    return df
# Define the function to apply moving sum operation
def to_movingsum(df, col, column_name):
    col_name = column_name[0]
    window = int(column_name[1])  # Convert window to an integer
    df[col] = df[col_name].rolling(window=window).sum()
    return df

def to_movingavg(df, col, column_name):
    col_name = column_name[0]
    window   = int(column_name[1])
    df[col] = df[col_name].rolling(window=window).mean()
    return df

# Define the function to apply power operation
def to_power(df, col, column_name):
    col_name = column_name[0]
    exponent   = int(column_name[1])
    df[col] = np.power(df[col_name], exponent)
    return df
# Define the function to generate random numbers
def to_rand(df, col, column_name):
    col_name = column_name[0]
    # Generate random values between 0 and 1 for the column
    df[col] = np.random.rand(len(df))
    return df

# # Define the function to apply base conversion operation
def convert_base(df, col, column_name):
    col_name = column_name[0]
    df[col] = df[col_name].apply(lambda x: np.base_repr(x,))
    return df

# def convert_base(df, column_name):
#     col_name = column_name[0]
#     input_base = int(column_name[1])
#     output_base= column_name[2]
#     df[col_name] = df[col_name].apply(lambda x: np.base_repr(int(x, input_base), output_base))
#     return df

############DATE####################################################################################################################
import pandas as pd
from datetime import datetime


def add_to_date(df, col, column_name):
    col_name = column_name[0]
    days_to_add = int(column_name[1])  # Convert to integer
    months_to_add = int(column_name[2])  # Convert to integer
    years_to_add = int(column_name[3])  # Convert to integer
    
    # Calculate the total number of days to add
    total_days = days_to_add + months_to_add * 30 + years_to_add * 365
    
    # Add the timedelta to each date element in the column
    df[col] = pd.to_datetime(df[col_name]) + pd.to_timedelta(total_days, unit='D') 
    return df

def date_compare(df, col, column_name):
    col_name = column_name[0]  # Name for the new column
    date1 = pd.to_datetime(df[column_name[0]])  # Extracting first date column
    date2 = pd.to_datetime(df[column_name[1]])  # Extracting second date column
    df[col] = (date1 < date2).astype(int) - (date1 > date2).astype(int)  # Comparing dates and storing result
    return df

def get_date_part(df, col, column_name):
    col_name = column_name[0] 
    # Convert column to datetime format
    df[col] = pd.to_datetime(df[col_name])
    time_period= column_name[1]
    # Dictionary to map part string to corresponding pandas datetime attribute
    part_map = {
        'year': pd.to_datetime(df[col_name]).dt.year,
        'month': pd.to_datetime(df[col_name]).dt.month,
        'day': pd.to_datetime(df[col_name]).dt.day,
        'hour': pd.to_datetime(df[col_name]).dt.hour,
        'minute': pd.to_datetime(df[col_name]).dt.minute,
        'second': pd.to_datetime(df[col_name]).dt.second
    }
    # Returning the specified part of the date column
    df[col] = part_map[time_period]
    return df



def rounddate(df, col, column_name):
    col_name = column_name[0]
    df[col] = pd.to_datetime(df[col_name]) 
    freq = column_name[1]
   
    if freq == 'YY':
        # Round year
        df[col] = df[col_name].apply(lambda x: x.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0))
    elif freq == 'MM':
        # Round month
        df[col] = df[col_name].apply(lambda x: x.replace(day=1, hour=0, minute=0, second=0, microsecond=0) if x.day <= 15 else (x.replace(day=1) + pd.DateOffset(months=1)).replace(hour=0, minute=0, second=0, microsecond=0))
    elif freq == 'DD':
        # Round day
        df[col] = df[col_name].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    elif freq == 'HH':
        # Round hour
        df[col] = df[col_name].apply(lambda x: x.replace(minute=0, second=0, microsecond=0) if x.minute < 30 else (x + pd.DateOffset(hours=1)).replace(minute=0, second=0, microsecond=0))
    elif freq == 'MI':
        # Round minute
        df[col] = df[col_name].apply(lambda x: x.replace(second=0, microsecond=0) if x.second < 30 else (x + pd.DateOffset(minutes=1)).replace(second=0, microsecond=0))
    elif freq == 'SS':
        # Round second
        df[col] = df[col_name].apply(lambda x: x.replace(microsecond=0) if x.microsecond < 500000 else (x + pd.DateOffset(seconds=1)).replace(microsecond=0))
    elif freq == 'MS':
        # Round millisecond
        df[col] = df[col_name].apply(lambda x: x.replace(microsecond=0) if x.microsecond < 500 else (x + pd.DateOffset(milliseconds=1)).replace(microsecond=0))
    elif freq == 'US':
        # Round microsecond
        df[col] = df[col_name].apply(lambda x: x.replace(nanosecond=0) if x.nanosecond < 500 else (x + pd.DateOffset(microseconds=1)).replace(nanosecond=0))
    
    return df


import calendar

def set_date_part(df, col, column_name):
    # Extract column name, part abbreviation, and value from the column_name tuple
    col_name = column_name[0]
    part = column_name[1]  # No need to remove quotes since it's already a string
    value =int(column_name[2])
    
    # Convert column to datetime if not already
    df[col_name] = pd.to_datetime(df[col_name])
    
    # Dictionary mapping parts to corresponding functions
    set_functions = {
        'YY': lambda x: x.replace(year=value),
        'MM': lambda x: pd.to_datetime(f"{x.year}-{value}-{min(x.day, calendar.monthrange(x.year, int(value))[1])}"),
        'DD': lambda x: x.replace(day=min(value, calendar.monthrange(x.year, x.month)[1])),
        'HH': lambda x: x.replace(hour=value),
        'MI': lambda x: x.replace(minute=value),
        'SS': lambda x: x.replace(second=value),
        'MS': lambda x: x.replace(microsecond=value * 1000),  # Milliseconds
        'US': lambda x: x.replace(microsecond=value)  # Microseconds
    }
    df[col] = df[col_name].apply(set_functions[part])
    return df


def systimestamp(df, col, column_name):
    col_name = column_name[0]
    system_timestamp = datetime.now()
    df[col_name] = pd.to_datetime(df[col_name])  # Convert the existing column to datetime if needed
    # Assign system timestamp to the specified column
    df[col] = system_timestamp
    return df

def chardate(df, col, column_name):

    # Extract column name from the column_name tuple
    col_name = column_name[0]
    format_string = column_name[1]
    # Convert column to datetime if not already
    df[col_name] = pd.to_datetime(df[col_name])
    # Apply the format string using strftime
    df[col] = df[col_name].dt.strftime(format_string)
    return df

def truncdate(df, col, column_name):
    # Extract column name from the column_name tuple
    col_name = column_name[0]
    time_period = column_name[1]
    # Convert column to datetime if not already
    df[col_name] = pd.to_datetime(df[col_name])
    # Dictionary mapping parts to corresponding functions
    trunc_functions = {
        'YY': lambda x: x.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0),
        'MM': lambda x: x.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        'DD': lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0),
        'HH': lambda x: x.replace(minute=0, second=0, microsecond=0),
        'MI': lambda x: x.replace(second=0, microsecond=0),
        'SS': lambda x: x.replace(microsecond=0),
        'MS': lambda x: x.replace(microsecond=x.microsecond // 1000 * 1000),  # Milliseconds
        'US': lambda x: x  # No truncation for microseconds
    }
    df[col] = df[col_name].apply(trunc_functions[time_period])
    return df
# Define the function to calculate the difference between two dates
def date_diff(df, col, column_name):
    col_name = column_name[0]  # Name for the new column
    date1 = pd.to_datetime(df[column_name[0]])  # Extracting first date column
    date2 = pd.to_datetime(df[column_name[1]])  # Extracting second date column
    difference_type = column_name[2]  # Extracting difference type
    
    # Calculating differences based on difference_type
    if difference_type == 'day':
        df[col] = (date2 - date1).dt.days
    elif difference_type == 'month':
        df[col] = (date2.dt.month - date1.dt.month) + 12 * (date2.dt.year - date1.dt.year)
    elif difference_type == 'year':
        df[col] = date2.dt.year - date1.dt.year
    else:
        raise ValueError("Invalid difference_type. Please choose 'day', 'month', or 'year'.")
    
    return df

# Define the function to get the last day of the month with time
def last_day(df, col, column_name):
    col_name = column_name[0]
    # Convert column to datetime format
    df[col] = pd.to_datetime(df[col_name])
    # Get the last day of each month
    df[col] = df[col] + pd.offsets.MonthEnd(0)
    return df

def make_date_time(df, col, column_name):
    col_name = column_name[0] 
    df[col] = pd.to_datetime(df[col_name])
    # Get the current system time
    current_time = datetime.now()
    df[col] = current_time
    return df

# exp_dict = {"upper(": to_upper, "lower(": to_lower, "reverse(": to_reverse, "ascii(": to_ascii, "substr(": to_substr, "rtrim(": to_rtrim, "ltrim(": to_ltrim, "rpad(": to_rpad, "replacestr(": to_replacestr, "replacechr(": to_replacechr, "lpad(": to_lpad, "length(": to_length, "instr(": to_instr, "chr(": to_chr, "chrcode(": to_chrcode, "concat(": to_concat, "indexof(": to_indexof, "initcap(": to_initcap,"abs(": to_abs, "ceil(": to_ceil, "convert_base(": convert_base, "cume(": to_cume, "exp(": to_exp, "floor(": to_floor, "ln(": to_ln, "round(": to_round, "sign(": to_sign, "sqrt(": to_sqrt, "trunc(": to_trunc,"log(": to_log, "mod(": to_mod, "movingsum(": to_movingsum, "power(": to_power, "movingavg(": to_movingavg, "rand(": to_rand, "add_to_date(": add_to_date, "date_compare(": date_compare, "date_diff(" : date_diff, "get_date_part(": get_date_part, "last_day(" : last_day, 'rounddate(':rounddate, "set_date_part(":set_date_part, "truncdate(": truncdate, "chardate(":chardate, "systimestamp(": systimestamp}

####################################Conversion#########################################
from decimal import Decimal

def to_bigint(df, col, column_name):
    col_name = column_name[0]
    df[col] = df[col_name].astype('int32')
    return df

def to_char(df, col, column_name):
    col_name = column_name[0]
    decimal_places=2
    df[col] = df[col_name].apply(lambda x: '{:.{}f}'.format(x, decimal_places))
    return df

def to_date(df, col, column_name):
    # Extract column name from the column_name tuple
    col_name = column_name[0].strip()
    format_string = column_name[1]
    # Convert column to datetime if not already
    df[col_name] = pd.to_datetime(df[col_name])
    # Apply the format string using strftime
    df[col] = df[col_name].dt.strftime(format_string)
    return df

# Define the function to convert a column to Decimal
def to_decimal(df, col, column_name):
    col_name = column_name[0]
    scale= int(column_name[1]) 
    df[col] = df[col_name].apply(lambda x: Decimal(x).quantize(Decimal('0.' + '0'*scale)))
    return df

# Define the function to convert a column to float
def to_float(df, col, column_name):
    col_name = column_name[0]
    df[col] = pd.to_numeric(df[col_name], errors='coerce').astype(float)
    return df

# Define the function to convert a column to Integer
def to_integer(df, col, column_name):
    col_name = column_name[0]
    flag= int(column_name[1])
    if flag==1:
        df[col] = df[col_name].apply(lambda x: int(x))
    elif flag==0:
        df[col] = df[col_name].apply(lambda x: round(float(x)))
    return df

import pandas as pd

#exp_dict = {"to_bigint(": to_bigint, "to_char(":to_char, "to_decimal(": to_decimal, "to_float(": to_float, "to_integer(": to_integer, "to_date(":to_date}


##################################### Data_Cleansing ###################################################################################################

# def greatest(df, col, column_name):
#     col_name = column_name[0]
#     max_value = df[col_name].max()
#     max_index = df[col_name].idxmax()
#     df[col] = np.where(df.index == max_index, max_value, np.nan)
#     return df

def greatest(df, col, column_name):
    col_name = column_name[0]  # Assuming you only want to consider the first column name in the list
    max_value = df[col_name].max()
    df[col] = df[col].apply(lambda x: 0 if x != max_value else x)
    return df

def is_in(df, col, column_name):
    col_name = column_name[0]
    df[col] = chunk[col_name].isin(chunk[column_name[1]])
    return chunk



#exp_dict = {"greatest(": greatest,"is_in(": is_in}

##################################### Test ###################################################################################################

def is_date(df, col, column_name):
    col_name = column_name[0]
    column_string = df[col_name].astype(str)
    format = "%Y-%m-%d" or '%d-%m-%Y'or '%m-%d-%Y' # Default date format, change as needed
    # Check if each value in the column is a valid date
    df[col] = column_string.apply(lambda x: True if len(x.strip()) == len(format) else False)
    return df
exp_dict= {"is_date(": is_date}