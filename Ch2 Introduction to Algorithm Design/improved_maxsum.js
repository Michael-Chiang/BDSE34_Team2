function maxSum(arr, size) {
  if (size > arr.length) {
    return null;
  }

  let maxValue = 0;
  for (let i = 0; i < size; i++) {
    maxValue += arr[i];
  }

  let temValue = maxValue;

  for (let j = size; j < arr.length; j++) {
    temValue = temValue + arr[j] - arr[j - size];
    if (temValue > maxValue) {
      maxValue = temValue;
    }
  }

  return maxValue;
}

console.log(maxSum([2, 7, 3, 7, 25, 0, 6, 1, -5, -12, -11], 1));
