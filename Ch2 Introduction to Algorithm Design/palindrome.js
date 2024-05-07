function palindrome(str) {
  let left = 0;
  let right = str.length - 1;
  while (left <= right) {
    if (str[left] == str[right]) {
      left++;
      right--;
    } else {
      console.log(false);
      return false;
    }
  }
  console.log(true);
  return true;
}

palindrome("tacocat");
palindrome("saippuakivikauppias");
palindrome("saippuakivikauppia");
