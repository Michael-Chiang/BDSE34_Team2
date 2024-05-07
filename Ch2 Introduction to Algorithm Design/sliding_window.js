function maxSum(arr, size) {
  let max_value = -Infinity;
  if (size > arr.length) {
    return null;
  }

  for (let i = 0; i <= arr.length - size; i++) {
    let attempt = 0;
    for (let j = i; j < i + size; j++) {
      attempt += arr[j];
    }
    if (attempt > max_value) {
      max_value = attempt;
    }
  }
  console.log(max_value);
  return max_value;
}

function minSum(arr, size) {
  let min_value = Infinity;
  if (size > arr.length) {
    return null;
  }

  for (let i = 0; i <= arr.length - size; i++) {
    let attempt = 0;
    for (let j = i; j < i + size; j++) {
      attempt += arr[j];
    }
    if (attempt < min_value) {
      min_value = attempt;
    }
  }
  console.log(min_value);
  return min_value;
}

maxSum([2, 7, 3, 0, 6, 1, -5, -12, -11], 2);
minSum([2, 7, -30, 0, 6, 1, -5, -12, -11], 2);
