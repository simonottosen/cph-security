create table if not exists waitingtime (
  id bigint generated by default as identity primary key,
  queue bigint,
  timestamp timestamp with time zone default timezone('cet'::text, now()) not null,
  airport text
);
