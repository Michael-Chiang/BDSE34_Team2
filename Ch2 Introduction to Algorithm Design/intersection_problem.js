function intersection(arr1, arr2) {
  let result = [];
  for (let i = 0; i < arr1.length; i++) {
    for (let j = 0; j < arr2.length; j++) {
      console.log(arr1[i], arr2[j]);
      if (arr1[i] == arr2[j]) {
        result.push(arr1[i]);
      }
    }
  }
  console.log(result);
  return result;
}

intersection([1, 2, 3, 7, 9, 19, 25], [19, 5, 16, 10, 3, 1]);
