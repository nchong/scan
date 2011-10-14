#ifndef SCANREF_H
#define SCANREF_H

/*
 * Exclusive scan on array [input] of length [n]
 */
void exclusive_scan_host(int *output, int *input, int n);

/*
 * Segmented exclusive scan on array tuple ([input], [flag]), both of length [n]
 */
void segmented_exclusive_scan_host(int *output, int *input, int *flag, int n);

#endif
