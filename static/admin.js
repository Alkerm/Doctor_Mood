const adminKeyInput = document.getElementById('admin-key');
const loadUsersBtn = document.getElementById('load-users-btn');
const refreshUsersBtn = document.getElementById('refresh-users-btn');
const exportUsersBtn = document.getElementById('export-users-btn');
const adminMessage = document.getElementById('admin-message');
const usersTbody = document.getElementById('users-tbody');
const adminAddBox = document.getElementById('admin-add-box');
const targetEmailInput = document.getElementById('target-email');
const addUsesInput = document.getElementById('add-uses');
const addTrialsBtn = document.getElementById('add-trials-btn');

function setAdminMessage(message, isError = false) {
    adminMessage.textContent = message || '';
    adminMessage.style.color = isError ? '#b91c1c' : '#374151';
}

function adminHeaders() {
    return {
        'Content-Type': 'application/json',
        'X-Admin-Key': (adminKeyInput.value || '').trim()
    };
}

function renderUsers(users) {
    usersTbody.innerHTML = '';
    users.forEach(user => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${user.id}</td>
            <td>${user.email}</td>
            <td>${user.used_uses}</td>
            <td>${user.total_uses}</td>
            <td>${user.remaining_uses}</td>
            <td>${user.created_at || ''}</td>
            <td>${user.last_login_at || ''}</td>
        `;
        usersTbody.appendChild(row);
    });
}

function setAdminAuthenticated(isAuthenticated) {
    adminAddBox.style.display = isAuthenticated ? 'grid' : 'none';
}

async function loadUsers() {
    const key = (adminKeyInput.value || '').trim();
    if (!key) {
        setAdminMessage('Enter admin key first.', true);
        return;
    }

    setAdminMessage('Loading users...');
    try {
        const response = await fetch('/admin/users', {
            method: 'GET',
            headers: adminHeaders()
        });
        const data = await response.json();
        if (!response.ok) {
            setAdminAuthenticated(false);
            throw new Error(data.error || `Failed (HTTP ${response.status})`);
        }
        renderUsers(data.users || []);
        setAdminAuthenticated(true);
        setAdminMessage(`Loaded ${data.count || 0} users.`);
    } catch (error) {
        setAdminAuthenticated(false);
        setAdminMessage(error.message || 'Failed to load users.', true);
    }
}

async function addTrials() {
    const key = (adminKeyInput.value || '').trim();
    const email = (targetEmailInput.value || '').trim();
    const additionalUses = Number(addUsesInput.value);

    if (!key) {
        setAdminMessage('Enter admin key first.', true);
        setAdminAuthenticated(false);
        return;
    }
    if (!email) {
        setAdminMessage('Enter user email.', true);
        return;
    }
    if (!Number.isInteger(additionalUses) || additionalUses === 0) {
        setAdminMessage('Enter a non-zero integer (positive to add trials, negative to reduce).', true);
        return;
    }

    setAdminMessage('Adding trials...');
    try {
        const response = await fetch('/admin/add-trials', {
            method: 'POST',
            headers: adminHeaders(),
            body: JSON.stringify({
                email,
                additional_uses: additionalUses
            })
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || `Failed (HTTP ${response.status})`);
        }
        setAdminMessage(data.message || 'Trials added.');
        await loadUsers();
    } catch (error) {
        setAdminMessage(error.message || 'Failed to add trials.', true);
    }
}

function exportUsersCsv() {
    const key = (adminKeyInput.value || '').trim();
    if (!key) {
        setAdminMessage('Enter admin key first.', true);
        setAdminAuthenticated(false);
        return;
    }

    const url = `/admin/export-users?ts=${Date.now()}`;
    fetch(url, {
        method: 'GET',
        headers: { 'X-Admin-Key': key }
    }).then(async (response) => {
        if (!response.ok) {
            let err = 'Export failed';
            try {
                const data = await response.json();
                err = data.error || err;
            } catch (_) {
                // ignore parse error
            }
            setAdminAuthenticated(false);
            throw new Error(err);
        }

        setAdminAuthenticated(true);
        const blob = await response.blob();
        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        const contentDisposition = response.headers.get('Content-Disposition') || '';
        const match = contentDisposition.match(/filename=([^;]+)/);
        const fileName = match ? match[1] : `users_export_${Date.now()}.csv`;
        link.href = downloadUrl;
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(downloadUrl);
        setAdminMessage('CSV export downloaded.');
    }).catch((error) => {
        setAdminMessage(error.message || 'Export failed.', true);
    });
}

loadUsersBtn.addEventListener('click', loadUsers);
refreshUsersBtn.addEventListener('click', loadUsers);
addTrialsBtn.addEventListener('click', addTrials);
exportUsersBtn.addEventListener('click', exportUsersCsv);
setAdminAuthenticated(false);
