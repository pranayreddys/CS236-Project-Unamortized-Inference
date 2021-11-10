def find_max_students(n, m, edges, s):
	
	Components = find_SCCs(n, m, edges)
	max_students = 0
	for component in Components:
		students[component] = 0
		for node in component:
			students[component] += s[node]
	ord = TopologicalSort(Components.size(), CompEdges.size(), CompEdges)
	ans = 0
	DP = [0] * n
	for j in ord:
		for k in adjacent[j]:
			DP[k] = max(DP[k], DP[j] + students[k])
		ans = max(ans, DP[j])

	return ans