import { useState } from 'react'

function App() {
  const [msg, setMsg] = useState('')

  async function createUser() {
    const res = await fetch('http://localhost:8000/users?username=test&api_key=key', { method: 'POST' })
    const data = await res.json()
    setMsg(`Created user with ID ${data.id}`)
  }

  return (
    <div>
      <h1>Datacreek</h1>
      <button onClick={createUser}>Create Test User</button>
      <p>{msg}</p>
    </div>
  )
}

export default App
