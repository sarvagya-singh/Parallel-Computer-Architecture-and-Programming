#include <stdio.h>
#include <cuda_runtime.h>

#define ITEMS 5
#define FRIENDS 3

struct Item {
    char name[20];
    int price;
};

__global__ void calculateTotal(int *purchases, int *prices, int *totals, int numItems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < FRIENDS) {
        int total = 0;
        for (int i = 0; i < numItems; i++) {
            total += purchases[idx * numItems + i] * prices[i];
        }
        totals[idx] = total;
    }
}

int main() {
    Item items[ITEMS] = {
        {"Shoes", 50},
        {"Shirt", 30},
        {"Bag", 40},
        {"Watch", 100},
        {"Sunglasses", 25}
    };

    int purchases[FRIENDS][ITEMS] = {
        {1, 2, 0, 1, 1},
        {0, 1, 1, 0, 2},
        {2, 0, 1, 1, 0} 
    };
    
    int prices[ITEMS], totals[FRIENDS];
    for (int i = 0; i < ITEMS; i++) {
        prices[i] = items[i].price;
    }

    int *d_purchases, *d_prices, *d_totals;
    cudaMalloc(&d_purchases, FRIENDS * ITEMS * sizeof(int));
    cudaMalloc(&d_prices, ITEMS * sizeof(int));
    cudaMalloc(&d_totals, FRIENDS * sizeof(int));

    cudaMemcpy(d_purchases, purchases, FRIENDS * ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prices, prices, ITEMS * sizeof(int), cudaMemcpyHostToDevice);

    calculateTotal<<<1, FRIENDS>>>(d_purchases, d_prices, d_totals, ITEMS);
    cudaMemcpy(totals, d_totals, FRIENDS * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nShopping Mall Menu:\n");
    for (int i = 0; i < ITEMS; i++) {
        printf("%d. %s - $%d\n", i + 1, items[i].name, items[i].price);
    }
    for (int i = 0; i < FRIENDS; i++) {
        printf("\nFriend %d purchased:\n", i + 1);
        for (int j = 0; j < ITEMS; j++) {
            if (purchases[i][j] > 0)
                printf("  %d x %s\n", purchases[i][j], items[j].name);
        }
        printf("Total: $%d\n", totals[i]);
    }

    cudaFree(d_purchases);
    cudaFree(d_prices);
    cudaFree(d_totals);

    return 0;
}
