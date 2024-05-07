function insertionSort(arr) {
  for (let j = 1; j <= arr.length - 1; j++) {
    key = arr[j];
    let i = j - 1;
    while (i >= 0 && arr[i] > key) {
      arr[i + 1] = arr[i];
      i--;
    }
    arr[i + 1] = key;
  }
  console.log(arr);
  return arr;
}

let unsorted = [14, -4, 17, 6, 22, 1, -5];
insertionSort(unsorted);
insertionSort([5, 4, 3, 2, 1]);
