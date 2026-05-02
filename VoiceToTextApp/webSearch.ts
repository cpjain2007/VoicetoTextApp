import { Alert, Linking } from "react-native";

export function openGoogleSearch(query: string): void {
  const q = query.trim();
  if (!q) {
    Alert.alert("Empty search", "Type something to search the web.");
    return;
  }
  const url = `https://www.google.com/search?q=${encodeURIComponent(q)}`;
  void Linking.openURL(url).catch(() => {
    Alert.alert("Could not open browser", "Try again or search manually in your browser.");
  });
}
