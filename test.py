
if __name__ == "__main__":
    lags = [[1, 4, 12, 24, 48]]
    output_lags = list([int(lag) for sub_list in lags for lag in sub_list])
    print(output_lags)
    output_lags = sorted(list(set(output_lags)))
    print(output_lags)