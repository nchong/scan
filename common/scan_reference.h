/*
 * Exclusive scan on array [input] of length [n]
 */
void exclusive_scan_host(int *output, int *input, int n) {
  output[0] = 0;
  for (int i=1; i<n; i++) {
    output[i] = output[i-1] + input[i-1];
  }
}

/*
 * Segmented exclusive scan on array tuple ([input], [flag]), both of length [n]
 */
void segmented_exclusive_scan_host(int *output, int *input, int *flag, int n) {
  output[0] = 0;
  for (int i=1; i<n; i++) {
    if (flag[i]) {
      output[i] = 0;
    } else {
      output[i] = output[i-1] + input[i-1];
    }
  }
}
